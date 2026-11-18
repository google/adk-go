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
	"context"
	"errors"
	"fmt"
	"iter"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// scriptedModel answers call i with turns[i] (the last entry once the script
// runs out), one LLMResponse per element, and counts its calls.
func scriptedModel(calls *int, turns ...[]*model.LLMResponse) *mockModelForTest {
	return &mockModelForTest{
		name: "scripted",
		generateContent: func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
			i := min(*calls, len(turns)-1)
			*calls++
			return func(yield func(*model.LLMResponse, error) bool) {
				for _, r := range turns[i] {
					if !yield(r, nil) {
						return
					}
				}
			}
		},
	}
}

func newTool(t *testing.T, name string, run func(agent.Context) (map[string]any, error)) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name}, func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		return run(ctx)
	})
	if err != nil {
		t.Fatalf("functiontool.New(%q) error = %v", name, err)
	}
	return tl
}

func textResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel)}
}

func callResponse(id, name string) *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: map[string]any{}}}},
	}}
}

// abortableInvocation returns an invocation context whose run carries an
// abort state for signal, as the runner builds it: the state on the run
// context and, below it, the context the abort cancels.
func abortableInvocation(t *testing.T, signal <-chan struct{}) agent.InvocationContext {
	t.Helper()
	state := icontext.NewAbortState(context.Background(), signal)
	runCtx := icontext.WithAbortState(runconfig.ToContext(t.Context(), &runconfig.RunConfig{}), state)
	engine, stop := state.Watch(runCtx)
	t.Cleanup(stop)
	return icontext.NewInvocationContext(engine, icontext.InvocationContextParams{
		InvocationID: "inv_1",
		Agent:        &mockAgent{name: "agent_1"},
	})
}

// eventSummary renders an event as its text or its first function call or
// response, so a test can compare the yielded stream with a literal.
func eventSummary(ev *session.Event) string {
	if ev.Content == nil || len(ev.Content.Parts) == 0 {
		return "<no content>"
	}
	p := ev.Content.Parts[0]
	switch {
	case p.FunctionCall != nil:
		return "call:" + p.FunctionCall.Name
	case p.FunctionResponse != nil:
		return fmt.Sprintf("response:%s %v", p.FunctionResponse.Name, p.FunctionResponse.Response)
	default:
		return p.Text
	}
}

func TestFlowRun_Abort_SignalBeforeRunCallsNoModel(t *testing.T) {
	signal := make(chan struct{})
	close(signal)
	var modelCalls int
	f := &Flow{Model: scriptedModel(&modelCalls, []*model.LLMResponse{textResponse("hello")})}

	var got []string
	for ev, err := range f.Run(abortableInvocation(t, signal)) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		got = append(got, eventSummary(ev))
	}

	if modelCalls != 0 {
		t.Errorf("model called %d times, want 0", modelCalls)
	}
	if len(got) != 0 {
		t.Errorf("Run() yielded %q, want no events", got)
	}
}

// One model call streams two responses; the caller aborts on the first. The
// flow must not turn the second into an event (u5785).
func TestFlowRun_Abort_SignalDuringStepStopsBeforeNextEvent(t *testing.T) {
	signal := make(chan struct{})
	var modelCalls int
	f := &Flow{Model: scriptedModel(&modelCalls, []*model.LLMResponse{textResponse("chunk 1"), textResponse("chunk 2")})}

	var got []string
	for ev, err := range f.Run(abortableInvocation(t, signal)) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		got = append(got, eventSummary(ev))
		if len(got) == 1 {
			close(signal)
		}
	}

	if diff := cmp.Diff([]string{"chunk 1"}, got); diff != "" {
		t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
	}
}

// The caller aborts on the function response; the flow must not start the
// next model step.
func TestFlowRun_Abort_SignalAfterToolResponseCallsModelOnce(t *testing.T) {
	signal := make(chan struct{})
	var modelCalls int
	f := &Flow{
		Model: scriptedModel(&modelCalls,
			[]*model.LLMResponse{callResponse("call_1", "quick")},
			[]*model.LLMResponse{textResponse("done")},
		),
		Tools: []tool.Tool{newTool(t, "quick", func(agent.Context) (map[string]any, error) {
			return map[string]any{"ok": true}, nil
		})},
	}

	var got []string
	for ev, err := range f.Run(abortableInvocation(t, signal)) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		got = append(got, eventSummary(ev))
		if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].FunctionResponse != nil {
			close(signal)
		}
	}

	if modelCalls != 1 {
		t.Errorf("model called %d times, want 1", modelCalls)
	}
	if diff := cmp.Diff([]string{"call:quick", "response:quick map[ok:true]"}, got); diff != "" {
		t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
	}
}

// A tool running when the abort fires sees its context canceled at once, and
// its canceled result never becomes an event: the call stays open so the
// runner seals it with the abort message instead of the cancellation's text.
func TestFlowRun_Abort_RunningToolIsCanceledAndItsResponseDropped(t *testing.T) {
	signal := make(chan struct{})
	var modelCalls int
	started := make(chan struct{})
	toolErr := make(chan error, 1)
	f := &Flow{
		Model: scriptedModel(&modelCalls,
			[]*model.LLMResponse{callResponse("call_1", "_slow_tool")},
			[]*model.LLMResponse{textResponse("done")},
		),
		Tools: []tool.Tool{newTool(t, "_slow_tool", func(ctx agent.Context) (map[string]any, error) {
			close(started)
			select {
			case <-ctx.Done():
				toolErr <- ctx.Err()
				return nil, ctx.Err()
			case <-time.After(5 * time.Second):
				toolErr <- nil
				return map[string]any{"result": "finished"}, nil
			}
		})},
	}
	go func() {
		<-started
		close(signal)
	}()

	begin := time.Now()
	var got []string
	for ev, err := range f.Run(abortableInvocation(t, signal)) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		got = append(got, eventSummary(ev))
	}

	if elapsed := time.Since(begin); elapsed >= 4*time.Second {
		t.Errorf("Run() took %v, want well under the tool's 5s block", elapsed)
	}
	select {
	case err := <-toolErr:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("tool saw ctx.Err() = %v, want context.Canceled", err)
		}
	default:
		t.Error("the tool never ran")
	}
	if diff := cmp.Diff([]string{"call:_slow_tool"}, got); diff != "" {
		t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
	}
	if modelCalls != 1 {
		t.Errorf("model called %d times, want 1", modelCalls)
	}
}

// One model call streams a function call and then the text "after"; the
// abort fires while the tool runs. The flow must not go on to the call's
// next response once the abort has dropped the tool's response.
func TestFlowRun_Abort_ResponseAfterCanceledCallIsDropped(t *testing.T) {
	signal := make(chan struct{})
	var modelCalls int
	started := make(chan struct{})
	f := &Flow{
		Model: scriptedModel(&modelCalls,
			[]*model.LLMResponse{callResponse("call_1", "_slow_tool"), textResponse("after")},
			[]*model.LLMResponse{textResponse("second call")},
		),
		Tools: []tool.Tool{newTool(t, "_slow_tool", func(ctx agent.Context) (map[string]any, error) {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		})},
	}
	go func() {
		<-started
		close(signal)
	}()

	var got []string
	for ev, err := range f.Run(abortableInvocation(t, signal)) {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
		got = append(got, eventSummary(ev))
	}

	if diff := cmp.Diff([]string{"call:_slow_tool"}, got); diff != "" {
		t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
	}
	if modelCalls != 1 {
		t.Errorf("model called %d times, want 1", modelCalls)
	}
}
