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

package runner_test

import (
	"context"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// arSeal is the part of an event that the sealing of an aborted invocation
// sets. Role is "" for an event with no content.
type arSeal struct {
	InvocationID   string
	Author         string
	Branch         string
	IsolationScope string
	ErrorCode      string
	ErrorMessage   string
	Role           string
	Responses      []arResponse
}

type arResponse struct {
	ID       string
	Name     string
	Response map[string]any
}

func arSealOf(ev *session.Event) arSeal {
	s := arSeal{
		InvocationID:   ev.InvocationID,
		Author:         ev.Author,
		Branch:         ev.Branch,
		IsolationScope: ev.IsolationScope,
		ErrorCode:      ev.ErrorCode,
		ErrorMessage:   ev.ErrorMessage,
	}
	if ev.Content == nil {
		return s
	}
	s.Role = ev.Content.Role
	for _, p := range ev.Content.Parts {
		if fr := p.FunctionResponse; fr != nil {
			s.Responses = append(s.Responses, arResponse{ID: fr.ID, Name: fr.Name, Response: fr.Response})
		}
	}
	return s
}

func arSealsOf(evs []*session.Event) []arSeal {
	var out []arSeal
	for _, ev := range evs {
		out = append(out, arSealOf(ev))
	}
	return out
}

// arAbortedEvents returns the events of evs that carry the abort error code.
func arAbortedEvents(evs []*session.Event) []*session.Event {
	var out []*session.Event
	for _, ev := range evs {
		if ev.ErrorCode == "INVOCATION_ABORTED" {
			out = append(out, ev)
		}
	}
	return out
}

// arAbortedResponse is the response a seal gives each call it answers.
func arAbortedResponse() map[string]any {
	return map[string]any{"error": "Invocation was aborted by client."}
}

// arClosed returns an abort signal that is already closed.
func arClosed() chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

func arHasCall(ev *session.Event) bool {
	if ev.Content == nil {
		return false
	}
	for _, p := range ev.Content.Parts {
		if p.FunctionCall != nil {
			return true
		}
	}
	return false
}

// arRunAbortingAtCall ranges a run, closes abort while it handles the first
// event that carries a function call, as a caller that cancels at a call
// does, and keeps ranging to the end. It fails the test on an error.
func arRunAbortingAtCall(t *testing.T, r *runner.Runner, msg *genai.Content, abort chan struct{}, opts ...runner.RunOption) []*session.Event {
	t.Helper()
	opts = append(opts, runner.WithAbortSignal(abort))
	var events []*session.Event
	closed := false
	for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, msg, agent.RunConfig{}, opts...) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		events = append(events, ev)
		if !closed && arHasCall(ev) {
			close(abort)
			closed = true
		}
	}
	return events
}

// arCallingAgent returns a custom agent that counts its runs and, on its
// n-th run (from 1), yields one event holding calls(n) and then waits for
// its context to end, as an agent waiting on a slow tool does.
func arCallingAgent(t *testing.T, name string, runs *atomic.Int32, calls func(n int32) []*genai.FunctionCall, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:      name,
		SubAgents: subAgents,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				n := runs.Add(1)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.Branch = ctx.Branch()
				ev.Content = &genai.Content{Role: genai.RoleModel}
				for _, fc := range calls(n) {
					ev.Content.Parts = append(ev.Content.Parts, &genai.Part{FunctionCall: fc})
				}
				if !yield(ev, nil) {
					return
				}
				<-ctx.Done()
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	return a
}

// arCountingAgent returns a custom agent that counts its runs and yields
// one text event.
func arCountingAgent(t *testing.T, name string, runs *atomic.Int32, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:      name,
		SubAgents: subAgents,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				runs.Add(1)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.Branch = ctx.Branch()
				ev.Content = modelText("working")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	return a
}

// arRecordingModel is a fake model.LLM that answers the n-th request with
// the n-th content (the last one once they run out) and records every
// request.
type arRecordingModel struct {
	mu        sync.Mutex
	responses []*genai.Content
	requests  []*model.LLMRequest
}

func (m *arRecordingModel) Name() string { return "recording" }

func (m *arRecordingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		i := min(len(m.requests), len(m.responses)-1)
		m.requests = append(m.requests, req)
		content := m.responses[i]
		m.mu.Unlock()
		yield(&model.LLMResponse{Content: content}, nil)
	}
}

func (m *arRecordingModel) recorded() []*model.LLMRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*model.LLMRequest(nil), m.requests...)
}

// arRequestResponses returns the function responses of every content of req.
func arRequestResponses(req *model.LLMRequest) []arResponse {
	var out []arResponse
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			if fr := p.FunctionResponse; fr != nil {
				out = append(out, arResponse{ID: fr.ID, Name: fr.Name, Response: fr.Response})
			}
		}
	}
	return out
}

func arTexts(evs []*session.Event) []string {
	var out []string
	for _, ev := range evs {
		if text := eventText(ev); text != "" {
			out = append(out, ev.Author+": "+text)
		}
	}
	return out
}

func arCalls(calls ...*genai.FunctionCall) func(int32) []*genai.FunctionCall {
	return func(int32) []*genai.FunctionCall { return calls }
}

func arCall(id, name string) *genai.FunctionCall {
	return &genai.FunctionCall{ID: id, Name: name, Args: map[string]any{}}
}

func arSeedCall(inv, author, branch, scope, id, name string) *session.Event {
	ev := resumeSeed(inv, author, fnCallContent(id, name, map[string]any{}))
	ev.Branch = branch
	ev.IsolationScope = scope
	return ev
}

func TestRun_AbortUnderCallerInvocationID(t *testing.T) {
	t.Run("caller_id_on_new_turn", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc)
		var runs atomic.Int32
		root := arCallingAgent(t, "agent", &runs, arCalls(arCall("call_1", "tool_call_1")))
		r := newResumeRunner(t, root, svc, false)

		got := arRunAbortingAtCall(t, r, userText("start"), make(chan struct{}), runner.WithInvocationID("inv_1"))

		if len(got) != 2 {
			t.Fatalf("Run() yielded %d events, want 2 (the call and the seal): %v", len(got), describeEvents(got))
		}
		want := arSeal{
			InvocationID: "inv_1",
			Author:       "agent",
			ErrorCode:    "INVOCATION_ABORTED",
			ErrorMessage: "Invocation was aborted by client.",
			Role:         "user",
			Responses:    []arResponse{{ID: "call_1", Name: "tool_call_1", Response: arAbortedResponse()}},
		}
		if diff := cmp.Diff(want, arSealOf(got[1])); diff != "" {
			t.Errorf("seal mismatch (-want +got):\n%s", diff)
		}
		stored := sessionEvents(t, svc)
		if diff := cmp.Diff([]string{"inv_1", "inv_1", "inv_1"}, invocationIDs(stored)); diff != "" {
			t.Errorf("stored invocation IDs mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("function_response_overrides_caller_id", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc,
			seedUser("inv_orig", "go"),
			arSeedCall("inv_orig", "agent", "", "", "call_0", "t0"),
			arSeedCall("inv_orig", "agent", "", "", "call_1", "tool_call_1"),
		)
		var runs atomic.Int32
		root := arCallingAgent(t, "agent", &runs, arCalls())
		r := newResumeRunner(t, root, svc, false)

		got, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession,
			responseContent("call_0", "t0", map[string]any{"ok": true}), agent.RunConfig{},
			runner.WithInvocationID("inv_caller_ignored"), runner.WithAbortSignal(arClosed())))

		if len(errs) != 0 {
			t.Fatalf("Run() errors = %v", errs)
		}
		want := []arSeal{{
			InvocationID: "inv_orig",
			Author:       "agent",
			ErrorCode:    "INVOCATION_ABORTED",
			ErrorMessage: "Invocation was aborted by client.",
			Role:         "user",
			Responses:    []arResponse{{ID: "call_1", Name: "tool_call_1", Response: arAbortedResponse()}},
		}}
		if diff := cmp.Diff(want, arSealsOf(got)); diff != "" {
			t.Errorf("yielded events mismatch (-want +got):\n%s", diff)
		}
		if n := runs.Load(); n != 0 {
			t.Errorf("agent ran %d times, want 0", n)
		}
	})
}

func TestRun_AbortSealsOnlyTheAbortedInvocation(t *testing.T) {
	oldCall, err := functiontool.New(functiontool.Config{Name: "tool_old_call", Description: "Starts a long operation.", IsLongRunning: true},
		func(agent.Context, struct{}) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	childStarted := make(chan struct{})
	slowChild, err := agent.New(agent.Config{
		Name:        "tool_call_1",
		Description: "Works for a long time.",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				close(childStarted)
				select {
				case <-ctx.Done():
				case <-time.After(5 * time.Second):
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	m := scripted(
		&genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: arCall("old_call", "tool_old_call")}}},
		&genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "call_1", Name: "tool_call_1", Args: map[string]any{"request": "go"},
		}}}},
	)
	root := scriptedLlmAgent(t, "agent", m, llmagent.Config{Tools: []tool.Tool{oldCall, agenttool.New(slowChild, nil)}})
	svc := session.InMemoryService()
	seedResumeSession(t, svc)
	r := newResumeRunner(t, root, svc, false)

	first := runTurn(t, r, userText("first"), runner.WithInvocationID("inv_0"))
	if diff := cmp.Diff([]string{"agent call tool_old_call"}, describeEvents(first)); diff != "" {
		t.Fatalf("first run events mismatch (-want +got):\n%s", diff)
	}

	abort := make(chan struct{})
	go func() {
		select {
		case <-childStarted:
			close(abort)
		case <-t.Context().Done():
		}
	}()
	start := time.Now()
	got := runTurn(t, r, userText("second"), runner.WithInvocationID("inv_1"), runner.WithAbortSignal(abort))
	elapsed := time.Since(start)

	if elapsed >= 2*time.Second {
		t.Errorf("second run took %v, want the abort to stop the agent tool before its 5 s wait ends", elapsed)
	}
	if diff := cmp.Diff([]string{"agent call tool_call_1", "agent response tool_call_1"}, describeEvents(got)); diff != "" {
		t.Fatalf("second run events mismatch (-want +got):\n%s", diff)
	}
	want := arSeal{
		InvocationID: "inv_1",
		Author:       "agent",
		ErrorCode:    "INVOCATION_ABORTED",
		ErrorMessage: "Invocation was aborted by client.",
		Role:         "user",
		Responses:    []arResponse{{ID: "call_1", Name: "tool_call_1", Response: arAbortedResponse()}},
	}
	if diff := cmp.Diff(want, arSealOf(got[1])); diff != "" {
		t.Errorf("seal mismatch (-want +got):\n%s", diff)
	}
	responders := map[string]int{}
	for _, ev := range sessionEvents(t, svc) {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil {
				responders[p.FunctionResponse.ID]++
			}
		}
	}
	if diff := cmp.Diff(map[string]int{"call_1": 1}, responders); diff != "" {
		t.Errorf("stored function responses by call ID mismatch (-want +got):\n%s", diff)
	}
}

func TestRun_AbortedResumeSealsWholeInvocationInFirstSeenOrder(t *testing.T) {
	svc := session.InMemoryService()
	seedResumeSession(t, svc,
		seedUser("inv_1", "start"),
		arSeedCall("inv_1", "researcher", "root.researcher", "", "call_a", "tool_call_a"),
		arSeedCall("inv_1", "researcher", "root.researcher", "", "call_b", "tool_call_b"),
		arSeedCall("inv_1", "coder", "root.coder", "", "call_c", "tool_call_c"),
		arSeedCall("inv_1", "coder", "root.coder", "task_1", "call_d", "tool_call_d"),
	)
	var rootRuns atomic.Int32
	r := newResumeRunner(t, arCountingAgent(t, "root_agent", &rootRuns), svc, true)

	got := runTurn(t, r, nil, runner.WithInvocationID("inv_1"), runner.WithAbortSignal(arClosed()))

	seal := func(author, branch, scope string, calls ...string) arSeal {
		s := arSeal{
			InvocationID: "inv_1", Author: author, Branch: branch, IsolationScope: scope,
			ErrorCode: "INVOCATION_ABORTED", ErrorMessage: "Invocation was aborted by client.", Role: "user",
		}
		for _, id := range calls {
			s.Responses = append(s.Responses, arResponse{ID: id, Name: "tool_" + id, Response: arAbortedResponse()})
		}
		return s
	}
	want := []arSeal{
		seal("researcher", "root.researcher", "", "call_a", "call_b"),
		seal("coder", "root.coder", "", "call_c"),
		seal("coder", "root.coder", "task_1", "call_d"),
	}
	if diff := cmp.Diff(want, arSealsOf(got)); diff != "" {
		t.Errorf("yielded events mismatch (-want +got):\n%s", diff)
	}
	if n := rootRuns.Load(); n != 0 {
		t.Errorf("root agent ran %d times, want 0", n)
	}
	stored := sessionEvents(t, svc)
	if len(stored) != 8 {
		t.Fatalf("session holds %d events, want 8 (5 seeded, 3 seals): %v", len(stored), describeEvents(stored))
	}
	if diff := cmp.Diff(want, arSealsOf(stored[5:])); diff != "" {
		t.Errorf("stored events after the seed mismatch (-want +got):\n%s", diff)
	}
}

func TestRun_AbortThenResume_DoesNotRerunTool(t *testing.T) {
	tests := []struct {
		name  string
		calls []*genai.FunctionCall
		want  []arResponse
	}{
		{
			name:  "single_call_root",
			calls: []*genai.FunctionCall{arCall("call_1", "_counted_tool")},
			want:  []arResponse{{ID: "call_1", Name: "_counted_tool", Response: arAbortedResponse()}},
		},
		{
			name:  "two_calls_in_one_event",
			calls: []*genai.FunctionCall{arCall("call_1", "_counted_tool"), arCall("call_2", "_counted_tool")},
			want: []arResponse{
				{ID: "call_1", Name: "_counted_tool", Response: arAbortedResponse()},
				{ID: "call_2", Name: "_counted_tool", Response: arAbortedResponse()},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var toolRuns atomic.Int32
			counted, err := functiontool.New(functiontool.Config{Name: "_counted_tool", Description: "Counts its runs."},
				func(agent.Context, struct{}) (map[string]any, error) {
					toolRuns.Add(1)
					return map[string]any{"result": "ok"}, nil
				})
			if err != nil {
				t.Fatalf("functiontool.New() error = %v", err)
			}
			m := &arRecordingModel{responses: []*genai.Content{
				{Role: genai.RoleModel, Parts: func() []*genai.Part {
					var parts []*genai.Part
					for _, fc := range tc.calls {
						parts = append(parts, &genai.Part{FunctionCall: fc})
					}
					return parts
				}()},
				modelText("Recovered"),
			}}
			root := scriptedLlmAgent(t, "tool_agent", m, llmagent.Config{Tools: []tool.Tool{counted}})
			svc := session.InMemoryService()
			seedResumeSession(t, svc)
			r := newResumeRunner(t, root, svc, true)

			first := arRunAbortingAtCall(t, r, userText("Run"), make(chan struct{}))
			if diff := cmp.Diff([]string{"tool_agent call _counted_tool", "tool_agent response _counted_tool"}, describeEvents(first)); diff != "" {
				t.Fatalf("aborted run events mismatch (-want +got):\n%s", diff)
			}
			if first[1].InvocationID != first[0].InvocationID {
				t.Errorf("seal invocation ID = %q, want the call's %q", first[1].InvocationID, first[0].InvocationID)
			}

			resumed := runTurn(t, r, nil, runner.WithInvocationID(first[0].InvocationID))

			if n := toolRuns.Load(); n != 0 {
				t.Errorf("_counted_tool ran %d times, want 0", n)
			}
			if diff := cmp.Diff([]string{"tool_agent: Recovered"}, arTexts(resumed)); diff != "" {
				t.Errorf("resumed text events mismatch (-want +got):\n%s", diff)
			}
			reqs := m.recorded()
			if len(reqs) != 2 {
				t.Fatalf("model got %d requests, want 2", len(reqs))
			}
			if diff := cmp.Diff(tc.want, arRequestResponses(reqs[1])); diff != "" {
				t.Errorf("function responses of the resumed model request mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRun_AbortOnResumedSubAgentMarkerKeepsBranch(t *testing.T) {
	// The classic root is never an LlmAgent, so the history scan never picks
	// a sub-agent on a classic resume (a sub-agent must be transferable up
	// to the root). The resume reaches child through its answered call.
	seed := func(t *testing.T, svc session.Service) {
		seedResumeSession(t, svc,
			seedUser("inv_sub", "hello"),
			arSeedCall("inv_sub", "child", "root.child", "", "fc_x", "t"),
			func() *session.Event {
				ev := resumeSeed("inv_sub", "child", responseContent("fc_x", "t", map[string]any{"ok": true}))
				ev.Branch = "root.child"
				return ev
			}(),
		)
	}
	want := arSeal{
		InvocationID: "inv_sub",
		Author:       "root",
		Branch:       "root.child",
		ErrorCode:    "INVOCATION_ABORTED",
		ErrorMessage: "Invocation was aborted by client.",
	}

	t.Run("normal_path_pre_closed", func(t *testing.T) {
		svc := session.InMemoryService()
		seed(t, svc)
		var childRuns, rootRuns atomic.Int32
		child := arCountingAgent(t, "child", &childRuns)
		r := newResumeRunner(t, arCountingAgent(t, "root", &rootRuns, child), svc, true)

		got := runTurn(t, r, nil, runner.WithInvocationID("inv_sub"), runner.WithAbortSignal(arClosed()))

		if diff := cmp.Diff([]arSeal{want}, arSealsOf(got)); diff != "" {
			t.Errorf("yielded events mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("early_stop_defer", func(t *testing.T) {
		svc := session.InMemoryService()
		seed(t, svc)
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = modelText("step 2")
					if !yield(ev, nil) {
						return
					}
					<-ctx.Done()
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New() error = %v", err)
		}
		var rootRuns atomic.Int32
		r := newResumeRunner(t, arCountingAgent(t, "root", &rootRuns, child), svc, true)

		abort := make(chan struct{})
		for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, nil, agent.RunConfig{},
			runner.WithInvocationID("inv_sub"), runner.WithAbortSignal(abort)) {
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if eventText(ev) == "step 2" {
				close(abort)
				break
			}
		}

		stored := arAbortedEvents(sessionEvents(t, svc))
		if diff := cmp.Diff([]arSeal{want}, arSealsOf(stored)); diff != "" {
			t.Errorf("stored abort events mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestRun_AbortOnResumedNodeRunSealsUnderResumedID(t *testing.T) {
	svc := session.InMemoryService()
	seedResumeSession(t, svc,
		seedUser("inv_node", "start"),
		seedLongRunningCall("inv_node", "agent", "call_lr", "wait_tool"),
	)
	m := &arRecordingModel{responses: []*genai.Content{modelText("unexpected")}}
	r := newResumeRunner(t, scriptedLlmAgent(t, "agent", m, llmagent.Config{}), svc, true)

	got := runTurn(t, r, nil, runner.WithInvocationID("inv_node"), runner.WithAbortSignal(arClosed()))

	want := []arSeal{{
		InvocationID: "inv_node",
		Author:       "agent",
		ErrorCode:    "INVOCATION_ABORTED",
		ErrorMessage: "Invocation was aborted by client.",
		Role:         "user",
		Responses:    []arResponse{{ID: "call_lr", Name: "wait_tool", Response: arAbortedResponse()}},
	}}
	if diff := cmp.Diff(want, arSealsOf(got)); diff != "" {
		t.Errorf("yielded events mismatch (-want +got):\n%s", diff)
	}
	if n := len(m.recorded()); n != 0 {
		t.Errorf("model got %d requests, want 0", n)
	}
	if n := len(sessionEvents(t, svc)); n != 3 {
		t.Errorf("session holds %d events, want 3 (2 seeded, 1 seal)", n)
	}
}

func TestRun_ResumedInvocationAbortedAgainIsSealedAgain(t *testing.T) {
	svc := session.InMemoryService()
	seedResumeSession(t, svc)
	var runs atomic.Int32
	root := arCallingAgent(t, "agent", &runs, func(n int32) []*genai.FunctionCall {
		if n == 1 {
			return []*genai.FunctionCall{arCall("call_1", "t1")}
		}
		return []*genai.FunctionCall{arCall("call_2", "t2")}
	})
	r := newResumeRunner(t, root, svc, true)
	seal := func(inv string, responses ...arResponse) arSeal {
		s := arSeal{InvocationID: inv, Author: "agent", ErrorCode: "INVOCATION_ABORTED", ErrorMessage: "Invocation was aborted by client."}
		if len(responses) > 0 {
			s.Role = "user"
			s.Responses = responses
		}
		return s
	}

	round1 := arRunAbortingAtCall(t, r, userText("go"), make(chan struct{}))
	if len(round1) != 2 {
		t.Fatalf("round 1 yielded %d events, want 2: %v", len(round1), describeEvents(round1))
	}
	inv := round1[0].InvocationID
	if diff := cmp.Diff(seal(inv, arResponse{ID: "call_1", Name: "t1", Response: arAbortedResponse()}), arSealOf(round1[1])); diff != "" {
		t.Errorf("round 1 seal mismatch (-want +got):\n%s", diff)
	}

	round2 := arRunAbortingAtCall(t, r, nil, make(chan struct{}), runner.WithInvocationID(inv))
	if len(round2) != 2 {
		t.Fatalf("round 2 yielded %d events, want 2: %v", len(round2), describeEvents(round2))
	}
	if diff := cmp.Diff(seal(inv, arResponse{ID: "call_2", Name: "t2", Response: arAbortedResponse()}), arSealOf(round2[1])); diff != "" {
		t.Errorf("round 2 seal mismatch (-want +got):\n%s", diff)
	}

	round3 := runTurn(t, r, nil, runner.WithInvocationID(inv), runner.WithAbortSignal(arClosed()))
	if diff := cmp.Diff([]arSeal{seal(inv)}, arSealsOf(round3)); diff != "" {
		t.Errorf("round 3 events mismatch (-want +got):\n%s", diff)
	}
	if n := runs.Load(); n != 2 {
		t.Errorf("agent ran %d times, want 2 (not in round 3)", n)
	}
	if n := len(arAbortedEvents(sessionEvents(t, svc))); n != 3 {
		t.Errorf("session holds %d abort events, want 3", n)
	}
}

func TestRun_AbortOnEndedResumedAgentWritesNothing(t *testing.T) {
	t.Run("classic_ended_agent_writes_nothing", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc,
			seedUser("inv_ended", "hi"),
			seedText("inv_ended", "custom_root", "done"),
			seedEnd("inv_ended", "custom_root"),
		)
		var beforeRun, onEvent, afterRun atomic.Int32
		p, err := plugin.New(plugin.Config{
			Name: "counter",
			BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
				beforeRun.Add(1)
				return nil, nil
			},
			OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
				onEvent.Add(1)
				return ev, nil
			},
			AfterRunCallback: func(agent.InvocationContext) { afterRun.Add(1) },
		})
		if err != nil {
			t.Fatalf("plugin.New() error = %v", err)
		}
		var rootRuns atomic.Int32
		r := newResumeRunner(t, arCountingAgent(t, "custom_root", &rootRuns), svc, true, p)

		got := runTurn(t, r, nil, runner.WithInvocationID("inv_ended"), runner.WithAbortSignal(arClosed()))

		if len(got) != 0 {
			t.Errorf("Run() yielded %v, want nothing", describeEvents(got))
		}
		if n := len(sessionEvents(t, svc)); n != 3 {
			t.Errorf("session holds %d events, want the 3 seeded", n)
		}
		counts := []int32{beforeRun.Load(), onEvent.Load(), afterRun.Load(), rootRuns.Load()}
		if diff := cmp.Diff([]int32{0, 0, 0, 0}, counts); diff != "" {
			t.Errorf("before-run, on-event, after-run and agent run counts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("node_path_ended_agent_still_seals", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc,
			seedUser("inv_ended_node", "hi"),
			seedText("inv_ended_node", "llm_root", "done"),
			seedEnd("inv_ended_node", "llm_root"),
		)
		m := &arRecordingModel{responses: []*genai.Content{modelText("unexpected")}}
		r := newResumeRunner(t, scriptedLlmAgent(t, "llm_root", m, llmagent.Config{}), svc, true)

		got := runTurn(t, r, nil, runner.WithInvocationID("inv_ended_node"), runner.WithAbortSignal(arClosed()))

		want := []arSeal{{
			InvocationID: "inv_ended_node",
			Author:       "llm_root",
			ErrorCode:    "INVOCATION_ABORTED",
			ErrorMessage: "Invocation was aborted by client.",
		}}
		if diff := cmp.Diff(want, arSealsOf(got)); diff != "" {
			t.Errorf("yielded events mismatch (-want +got):\n%s", diff)
		}
		if n := len(sessionEvents(t, svc)); n != 4 {
			t.Errorf("session holds %d events, want 4 (3 seeded, 1 seal)", n)
		}
	})
}

func TestRun_AbortOnFailedInvocationResolutionSealsNothing(t *testing.T) {
	tests := []struct {
		name    string
		seed    []*session.Event
		msg     *genai.Content
		wantErr string
	}{
		{
			name:    "resumable_empty_session",
			wantErr: "has no events to resume",
		},
		{
			name:    "resumable_missing_user_message",
			seed:    []*session.Event{seedText("inv_0", "custom_root", "earlier")},
			wantErr: "no user message available for resuming invocation",
		},
		{
			name:    "strict_unmatched_function_response",
			seed:    []*session.Event{seedUser("inv_0", "hi")},
			msg:     responseContent("unknown_fc", "t", map[string]any{"ok": true}),
			wantErr: "function call not found for 1 function response ids",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, tc.seed...)
			var rootRuns atomic.Int32
			r := newResumeRunner(t, arCountingAgent(t, "custom_root", &rootRuns), svc, true)

			got, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, tc.msg, agent.RunConfig{},
				runner.WithInvocationID("inv_1"), runner.WithAbortSignal(arClosed())))

			if len(errs) != 1 || !strings.Contains(errs[0].Error(), tc.wantErr) {
				t.Fatalf("Run() errors = %v, want one containing %q", errs, tc.wantErr)
			}
			if len(got) != 0 {
				t.Errorf("Run() yielded %v, want nothing", describeEvents(got))
			}
			if n := len(sessionEvents(t, svc)); n != len(tc.seed) {
				t.Errorf("session holds %d events, want the %d seeded", n, len(tc.seed))
			}
		})
	}
}
