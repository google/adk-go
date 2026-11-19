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

package llminternal

import (
	"errors"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// stopCounts counts the tool callbacks a flow ran.
type stopCounts struct{ before, onError, after int }

// countingFlow returns a flow whose before-tool, on-tool-error and after-tool
// callbacks count their calls in c. onError is what each on-tool-error
// callback returns; afterSaw, when non-nil, receives the result the after-tool
// callback was handed.
func countingFlow(c *stopCounts, onError func() (map[string]any, error), afterSaw *map[string]any) *Flow {
	return &Flow{
		BeforeToolCallbacks: []BeforeToolCallback{func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
			c.before++
			return nil, nil
		}},
		OnToolErrorCallbacks: []OnToolErrorCallback{func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
			c.onError++
			return onError()
		}},
		AfterToolCallbacks: []AfterToolCallback{func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
			c.after++
			if afterSaw != nil {
				*afterSaw = result
			}
			return nil, nil
		}},
	}
}

func errorTool(name string, err error) *mockFunctionTool {
	return &mockFunctionTool{name: name, runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
		return nil, err
	}}
}

// erroringTool is a function tool, which the flow finds by its declaration,
// whose handler returns err.
func erroringTool(t *testing.T, name string, err error) tool.Tool {
	t.Helper()
	return newTool(t, name, func(agent.Context) (map[string]any, error) { return nil, err })
}

func stopTestContext(t *testing.T) agent.InvocationContext {
	t.Helper()
	return icontext.NewInvocationContext(
		runconfig.ToContext(t.Context(), &runconfig.RunConfig{}),
		icontext.InvocationContextParams{InvocationID: "inv_1", Agent: &mockAgent{name: "agent_1"}},
	)
}

// runFlowUntilError runs f and returns the yielded events and the first
// yielded error.
func runFlowUntilError(t *testing.T, f *Flow) ([]*session.Event, error) {
	t.Helper()
	var events []*session.Event
	for ev, err := range f.Run(stopTestContext(t)) {
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	return events, nil
}

func functionResponsesOf(events []*session.Event) []*genai.FunctionResponse {
	var out []*genai.FunctionResponse
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil {
				out = append(out, p.FunctionResponse)
			}
		}
	}
	return out
}

// Fails if a pause goes through the on-tool-error or the after-tool callbacks,
// or is answered instead of being carried out as the stop error.
func TestCallTool_PausedErrorStopsCall(t *testing.T) {
	var c stopCounts
	f := countingFlow(&c, func() (map[string]any, error) { return map[string]any{"handled": true}, nil }, nil)
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})

	res := f.callTool(agent.NewToolContext(ctx, "call_1", nil, nil), errorTool("ask", &toolinternal.PausedError{}), map[string]any{}, nil)

	if _, ok := errors.AsType[*toolinternal.PausedError](res.stop); !ok {
		t.Errorf("callTool() stop = %v, want a *toolinternal.PausedError", res.stop)
	}
	if diff := cmp.Diff(stopCounts{before: 1, onError: 0, after: 0}, c, cmp.AllowUnexported(stopCounts{})); diff != "" {
		t.Errorf("callback counts mismatch (-want +got):\n%s", diff)
	}
}

// Fails if a response is written for a sibling of a paused call: a partial
// response event breaks the replay of the paused call.
func TestHandleFunctionCalls_PauseReturnsErrorNoResponse(t *testing.T) {
	tools := map[string]tool.Tool{
		"a": errorTool("a", &toolinternal.PausedError{}),
		"b": &mockFunctionTool{name: "b", runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
			return map[string]any{"ok": 1}, nil
		}},
	}
	resp := pcCalls(
		&genai.FunctionCall{ID: "c1", Name: "a", Args: map[string]any{}},
		&genai.FunctionCall{ID: "c2", Name: "b", Args: map[string]any{}},
	)

	merged, err := (&Flow{}).handleFunctionCalls(stopTestContext(t), tools, resp, nil, nil)

	if _, ok := errors.AsType[*toolinternal.PausedError](err); !ok {
		t.Errorf("handleFunctionCalls() error = %v, want a *toolinternal.PausedError", err)
	}
	if merged != nil {
		t.Errorf("handleFunctionCalls() merged event = %+v, want nil", merged.Content)
	}
}

// Fails if the on-tool-error callbacks are skipped for a run-failing error, or
// if the after-tool callbacks run when the call stops the run.
func TestCallTool_FailRunErrorOffersCallbacksThenStops(t *testing.T) {
	failErr := &toolinternal.FailRunError{Err: errors.New("violation")}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})

	t.Run("callbacks decline", func(t *testing.T) {
		var c stopCounts
		f := countingFlow(&c, func() (map[string]any, error) { return nil, nil }, nil)

		res := f.callTool(agent.NewToolContext(ctx, "call_1", nil, nil), errorTool("gen", failErr), map[string]any{}, nil)

		if res.stop != failErr {
			t.Errorf("callTool() stop = %v, want the FailRunError", res.stop)
		}
		if diff := cmp.Diff(stopCounts{before: 1, onError: 1, after: 0}, c, cmp.AllowUnexported(stopCounts{})); diff != "" {
			t.Errorf("callback counts mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("callback answers", func(t *testing.T) {
		var c stopCounts
		var afterSaw map[string]any
		f := countingFlow(&c, func() (map[string]any, error) { return map[string]any{"a": 1}, nil }, &afterSaw)

		res := f.callTool(agent.NewToolContext(ctx, "call_1", nil, nil), errorTool("gen", failErr), map[string]any{}, nil)

		if res.stop != nil {
			t.Errorf("callTool() stop = %v, want nil", res.stop)
		}
		if diff := cmp.Diff(map[string]any{"a": 1}, afterSaw); diff != "" {
			t.Errorf("after-tool callback result mismatch (-want +got):\n%s", diff)
		}
	})
}

// Fails if the flow goes on to a second model call after a tool paused.
func TestFlowRun_PausedErrorEndsStepWithoutModelCall(t *testing.T) {
	var modelCalls int
	f := &Flow{
		Model: scriptedModel(&modelCalls,
			[]*model.LLMResponse{callResponse("call_1", "ask")},
			[]*model.LLMResponse{textResponse("done")},
		),
		Tools: []tool.Tool{erroringTool(t, "ask", &toolinternal.PausedError{})},
	}

	events, err := runFlowUntilError(t, f)

	if _, ok := errors.AsType[*toolinternal.PausedError](err); !ok {
		t.Errorf("Run() error = %v, want a *toolinternal.PausedError", err)
	}
	if modelCalls != 1 {
		t.Errorf("model called %d times, want 1", modelCalls)
	}
	if got := functionResponsesOf(events); len(got) != 0 {
		t.Errorf("Run() yielded function responses %v, want none", got)
	}
}

// Fails if an on-tool-error callback's own error for a run-failing error is
// answered to the model as {"error": "cb"} and the run goes on.
func TestCallTool_FailRunErrorCallbackError(t *testing.T) {
	var c stopCounts
	var modelCalls int
	f := countingFlow(&c, func() (map[string]any, error) { return nil, errors.New("cb") }, nil)
	f.Model = scriptedModel(&modelCalls,
		[]*model.LLMResponse{callResponse("call_1", "gen")},
		[]*model.LLMResponse{textResponse("done")},
	)
	f.Tools = []tool.Tool{erroringTool(t, "gen", &toolinternal.FailRunError{Err: errors.New("violation")})}

	events, err := runFlowUntilError(t, f)

	if err == nil || err.Error() != "cb" {
		t.Errorf("Run() error = %v, want the callback error %q", err, "cb")
	}
	if _, ok := errors.AsType[*toolinternal.FailRunError](err); ok {
		t.Errorf("Run() error = %v is the FailRunError, want the callback error", err)
	}
	if got := functionResponsesOf(events); len(got) != 0 {
		t.Errorf("Run() yielded function responses %v, want none", got)
	}
	if c.after != 0 {
		t.Errorf("after-tool callback ran %d times, want 0", c.after)
	}
	if modelCalls != 1 {
		t.Errorf("model called %d times, want 1", modelCalls)
	}
}

// Fails if a run-failing error ends the run although an on-tool-error
// callback answered it.
func TestCallTool_FailRunErrorAnsweredByCallback(t *testing.T) {
	var c stopCounts
	var modelCalls int
	var afterSaw map[string]any
	f := countingFlow(&c, func() (map[string]any, error) { return map[string]any{"a": 1}, nil }, &afterSaw)
	f.Model = scriptedModel(&modelCalls,
		[]*model.LLMResponse{callResponse("call_1", "gen")},
		[]*model.LLMResponse{textResponse("done")},
	)
	f.Tools = []tool.Tool{erroringTool(t, "gen", &toolinternal.FailRunError{Err: errors.New("violation")})}

	events, err := runFlowUntilError(t, f)
	if err != nil {
		t.Fatalf("Run() error = %v, want nil", err)
	}
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "gen", Response: map[string]any{"a": 1}}}
	if diff := cmp.Diff(want, functionResponsesOf(events)); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"a": 1}, afterSaw); diff != "" {
		t.Errorf("after-tool callback result mismatch (-want +got):\n%s", diff)
	}
	if modelCalls != 2 {
		t.Errorf("model called %d times, want 2", modelCalls)
	}
}

// Fails if the replay of a call reuses the approval the user gave the call
// before it paused: the body would run and its result would answer the call.
// adk-python's replay runs the calls without confirmations
// (flows/llm_flows/core/_function_call_postprocessor.py), so the gate asks
// again and the model is not called.
func TestFlowRun_ReplayedCallAsksForConfirmationAgain(t *testing.T) {
	var runs atomic.Int32
	ask, err := functiontool.New(functiontool.Config{Name: "ask", RequireConfirmation: true}, func(agent.Context, struct{}) (map[string]any, error) {
		runs.Add(1)
		return map[string]any{"region": "APAC"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	var modelCalls int
	ctx, sess := replayAgentContext(t, false, &mockAgent{name: "agent_1"})
	runReplayFlow(t, &Flow{Model: scriptedModel(&modelCalls, []*model.LLMResponse{callResponse("call_1", "ask")}), Tools: []tool.Tool{ask}}, ctx, sess)
	var confirmationID string
	for _, ev := range sess.events {
		for _, fc := range utils.FunctionCalls(utils.Content(ev)) {
			if fc.Name == toolconfirmation.FunctionCallName {
				confirmationID = fc.ID
			}
		}
	}
	if confirmationID == "" {
		t.Fatalf("first turn yielded no confirmation request; events: %v", describeReplayEvents(sess.events))
	}
	// The user approves, the call pauses for input on its own branch, and the
	// user answers there.
	approval := respEv("user", toolconfirmation.FunctionCallName, confirmationID)
	approval.Content.Parts[0].FunctionResponse.Response = map[string]any{"confirmed": true}
	answer := withBranch(respEv("user", requestInputFunctionCallName, "clarify_region"), "ask@call_1")
	answer.Content.Parts[0].FunctionResponse.Response = map[string]any{"result": "APAC"}
	for _, ev := range []*session.Event{
		approval,
		withBranch(withLRO(callEv("ask", fcall(requestInputFunctionCallName, "clarify_region")), "clarify_region"), "ask@call_1"),
		answer,
	} {
		ev.InvocationID = ctx.InvocationID()
		sess.events = append(sess.events, ev)
	}
	modelCalls = 0

	out := runReplayFlow(t, &Flow{Model: replayModel(t, &modelCalls), Tools: []tool.Tool{ask}}, ctx, sess)

	if got := runs.Load(); got != 0 {
		t.Errorf("handler runs = %d, want 0", got)
	}
	want := []*genai.FunctionResponse{{ID: "call_1", Name: "ask", Response: map[string]any{"error": `error tool "ask" requires confirmation, please approve or reject`}}}
	if diff := cmp.Diff(want, functionResponsesOf(out)); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
	var asked []string
	for _, ev := range out {
		if _, ok := ev.Actions.RequestedToolConfirmations["call_1"]; ok {
			asked = append(asked, "requested call_1")
		}
		for _, fc := range utils.FunctionCalls(utils.Content(ev)) {
			if fc.Name == toolconfirmation.FunctionCallName && fc.ID != confirmationID {
				asked = append(asked, "new confirmation call")
			}
		}
	}
	if diff := cmp.Diff([]string{"requested call_1", "new confirmation call"}, asked); diff != "" {
		t.Errorf("confirmation requests mismatch (-want +got):\n%s", diff)
	}
	if modelCalls != 0 {
		t.Errorf("model called %d times, want 0", modelCalls)
	}
}
