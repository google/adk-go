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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// abortTaskSetup is a coordinator in chat mode with a task-mode sub-agent,
// task_worker, whose one model response calls _slow_tool, a tool that waits
// five seconds or until its context ends.
type abortTaskSetup struct {
	svc          session.Service
	r            *runner.Runner
	coordModel   *arRecordingModel
	workerModel  *arRecordingModel
	toolStarted  chan struct{}
	startedOnce  sync.Once
	toolStarts   atomic.Int32
	toolFinishes atomic.Int32
}

func newAbortTaskSetup(t *testing.T) *abortTaskSetup {
	t.Helper()
	s := &abortTaskSetup{
		svc: session.InMemoryService(),
		coordModel: &arRecordingModel{responses: []*genai.Content{
			genai.NewContentFromFunctionCall("task_worker", map[string]any{"request": "look it up"}, genai.RoleModel),
			modelText("Coordinator reply"),
		}},
		workerModel: &arRecordingModel{responses: []*genai.Content{
			genai.NewContentFromFunctionCall("_slow_tool", map[string]any{}, genai.RoleModel),
		}},
		toolStarted: make(chan struct{}),
	}
	slowTool, err := functiontool.New(functiontool.Config{Name: "_slow_tool", Description: "waits five seconds"},
		func(ctx agent.Context, _ struct{}) (string, error) {
			s.toolStarts.Add(1)
			s.startedOnce.Do(func() { close(s.toolStarted) })
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(5 * time.Second):
				s.toolFinishes.Add(1)
				return "done", nil
			}
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	worker := scriptedLlmAgent(t, "task_worker", s.workerModel, llmagent.Config{
		Mode:  llmagent.ModeTask,
		Tools: []tool.Tool{slowTool},
	})
	coordinator := scriptedLlmAgent(t, "coordinator", s.coordModel, llmagent.Config{
		SubAgents: []agent.Agent{worker},
	})
	newNodeTestSession(t, t.Context(), s.svc)
	s.r = newNodeTestRunner(t, coordinator, s.svc)
	return s
}

// isSlowToolCall reports whether ev is task_worker's call of _slow_tool.
func isSlowToolCall(ev *session.Event) bool {
	return ev != nil && describeEvent(ev) == "task_worker call _slow_tool"
}

// firstCall returns the first function call of ev, or nil.
func firstCall(ev *session.Event) *genai.FunctionCall {
	if ev == nil || ev.Content == nil {
		return nil
	}
	for _, p := range ev.Content.Parts {
		if p.FunctionCall != nil {
			return p.FunctionCall
		}
	}
	return nil
}

// TestRunner_Abort_TaskSubAgentSealsBothScopesAndReleasesNextTurn pins an
// abort that reaches a task sub-agent while its tool is pending: the run
// stops without a tool response, a delegation response written by the
// coordinator, or an error; the seal answers the coordinator's delegation
// call (no scope) and task_worker's tool call (the delegation's scope); and
// the next message is not captured by the aborted task, so it reaches the
// coordinator (adk-python runners.py:126-127 and :148-152,
// events/_abort_events.py:106).
func TestRunner_Abort_TaskSubAgentSealsBothScopesAndReleasesNextTurn(t *testing.T) {
	const abortMsg = "Invocation was aborted by client."
	tests := []struct {
		name string
		// run ranges turn 1 with abort as the abort signal and returns the
		// events it read.
		run func(t *testing.T, s *abortTaskSetup, abort chan struct{}) []*session.Event
		// wantYielded is the number of turn-1 events the caller reads.
		wantYielded int
		// toolMustStart is set when the abort comes after _slow_tool started,
		// so the tool ran exactly once. When the abort comes inside the yield
		// of the call event, the tool may or may not start before the
		// cancellation reaches the task flow.
		toolMustStart bool
	}{
		{
			name: "close_in_yield_drain",
			run: func(t *testing.T, s *abortTaskSetup, abort chan struct{}) []*session.Event {
				var events []*session.Event
				for ev, err := range s.r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("Run task"), agent.RunConfig{}, runner.WithAbortSignal(abort)) {
					if err != nil {
						t.Fatalf("Run() error = %v", err)
					}
					events = append(events, ev)
					if isSlowToolCall(ev) {
						close(abort)
					}
				}
				return events
			},
			wantYielded: 4,
		},
		{
			name: "close_async_mid_tool_drain",
			run: func(t *testing.T, s *abortTaskSetup, abort chan struct{}) []*session.Event {
				var events []*session.Event
				var closer sync.WaitGroup
				runDone := make(chan struct{})
				defer func() {
					close(runDone)
					closer.Wait()
				}()
				for ev, err := range s.r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("Run task"), agent.RunConfig{}, runner.WithAbortSignal(abort)) {
					if err != nil {
						t.Fatalf("Run() error = %v", err)
					}
					events = append(events, ev)
					if isSlowToolCall(ev) {
						closer.Go(func() {
							select {
							case <-s.toolStarted:
								time.Sleep(20 * time.Millisecond)
								close(abort)
							case <-runDone:
							}
						})
					}
				}
				return events
			},
			wantYielded:   4,
			toolMustStart: true,
		},
		{
			name: "close_in_yield_break",
			run: func(t *testing.T, s *abortTaskSetup, abort chan struct{}) []*session.Event {
				var events []*session.Event
				for ev, err := range s.r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("Run task"), agent.RunConfig{}, runner.WithAbortSignal(abort)) {
					if err != nil {
						t.Fatalf("Run() error = %v", err)
					}
					events = append(events, ev)
					if isSlowToolCall(ev) {
						close(abort)
						break
					}
				}
				return events
			},
			wantYielded: 2,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := newAbortTaskSetup(t)

			start := time.Now()
			yielded := tc.run(t, s, make(chan struct{}))
			elapsed := time.Since(start)

			if elapsed >= time.Second {
				t.Errorf("turn 1 took %v, want under 1s (the tool waits 5s unless canceled)", elapsed)
			}
			if got := s.toolStarts.Load(); tc.toolMustStart && got != 1 {
				t.Errorf("_slow_tool started %d times, want 1", got)
			}
			if got := s.toolFinishes.Load(); got != 0 {
				t.Errorf("_slow_tool finished %d times, want 0", got)
			}
			stored := sessionEvents(t, s.svc)
			wantKinds := []string{
				"user text Run task",
				"coordinator call task_worker",
				"task_worker call _slow_tool",
				"coordinator response task_worker",
				"task_worker response _slow_tool",
			}
			if diff := cmp.Diff(wantKinds, describeEvents(stored)); diff != "" {
				t.Fatalf("turn 1 stored events mismatch (-want +got):\n%s", diff)
			}
			delegation, slowCall := firstCall(stored[1]), firstCall(stored[2])
			if delegation.ID == "" || slowCall.ID == "" {
				t.Fatalf("call ids = (%q, %q), want both set", delegation.ID, slowCall.ID)
			}
			inv := stored[1].InvocationID
			wantSeals := []arSeal{
				{InvocationID: inv, Author: "coordinator", Branch: stored[1].Branch, IsolationScope: "", Role: "model"},
				{InvocationID: inv, Author: "task_worker", Branch: stored[2].Branch, IsolationScope: delegation.ID, Role: "model"},
				{
					InvocationID: inv, Author: "coordinator", Branch: stored[1].Branch, IsolationScope: "",
					ErrorCode: "INVOCATION_ABORTED", ErrorMessage: abortMsg, Role: "user",
					Responses: []arResponse{{ID: delegation.ID, Name: "task_worker", Response: map[string]any{"error": abortMsg}}},
				},
				{
					InvocationID: inv, Author: "task_worker", Branch: stored[2].Branch, IsolationScope: delegation.ID,
					ErrorCode: "INVOCATION_ABORTED", ErrorMessage: abortMsg, Role: "user",
					Responses: []arResponse{{ID: slowCall.ID, Name: "_slow_tool", Response: map[string]any{"error": abortMsg}}},
				},
			}
			if diff := cmp.Diff(wantSeals, arSealsOf(stored[1:])); diff != "" {
				t.Errorf("turn 1 stored agent events mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(wantSeals[:tc.wantYielded], arSealsOf(yielded)); diff != "" {
				t.Errorf("turn 1 yielded events mismatch (-want +got):\n%s", diff)
			}

			turn2 := runTurn(t, s.r, userText("Follow up"))

			if diff := cmp.Diff([]string{"coordinator text Coordinator reply"}, describeEvents(turn2)); diff != "" {
				t.Fatalf("turn 2 yielded events mismatch (-want +got):\n%s", diff)
			}
			if got := turn2[0].IsolationScope; got != "" {
				t.Errorf("turn 2 reply IsolationScope = %q, want empty", got)
			}
			after := sessionEvents(t, s.svc)
			if got := describeEvent(after[len(stored)]); got != "user text Follow up" {
				t.Fatalf("first turn-2 stored event = %q, want %q", got, "user text Follow up")
			}
			if got := after[len(stored)].IsolationScope; got != "" {
				t.Errorf("stored Follow up IsolationScope = %q, want empty", got)
			}
			if got := len(s.workerModel.recorded()); got != 1 {
				t.Errorf("task_worker model got %d requests, want 1", got)
			}
			coordReqs := s.coordModel.recorded()
			if len(coordReqs) != 2 {
				t.Fatalf("coordinator model got %d requests, want 2", len(coordReqs))
			}
			// The request leaves out the client-generated "adk-" call id, as
			// adk-python's remove_client_function_call_id does.
			wantResponses := []arResponse{{ID: "", Name: "task_worker", Response: map[string]any{"error": abortMsg}}}
			if diff := cmp.Diff(wantResponses, arRequestResponses(coordReqs[1])); diff != "" {
				t.Errorf("coordinator request 1 function responses mismatch (-want +got):\n%s", diff)
			}
			contents := coordReqs[1].Contents
			if last := contents[len(contents)-1]; last.Role != genai.RoleUser || contentText(last) != "Follow up" {
				t.Errorf("coordinator request 1 last content = role %q text %q, want role %q text %q",
					last.Role, contentText(last), genai.RoleUser, "Follow up")
			}
		})
	}
}
