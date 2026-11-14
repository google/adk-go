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
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// A root graph workflow is a node in adk-python (workflow/_workflow.py), so
// Runner.run_async sends it to run_node_async, which refuses a message that
// mixes text with a function response that has an ID before it resolves the
// message's invocation (workflow/_node_runner_utils.py).
func TestRun_GraphWorkflowRoot_MixedMessageRefused(t *testing.T) {
	const wantErr = "message cannot contain both function responses and text: function responses resume an existing invocation while text starts a new one"
	seed := []*session.Event{
		seedUser("inv_a", "first"),
		resumeSeed("inv_a", "root", fnCallContent("fc-a", "tool", nil)),
	}
	mixed := func(id string) *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
			{Text: "hi"},
			{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "tool", Response: map[string]any{"ok": true}}},
		}}
	}
	tests := []iidRefusalCase{
		{name: "answers_stored_call", msg: mixed("fc-a")},
		{name: "answers_stored_call_resumable", resumable: true, msg: mixed("fc-a")},
		// Strict resolution would refuse the unknown call; the mixed message
		// is refused first.
		{name: "unknown_call_resumable", resumable: true, msg: mixed("fc-unknown")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.root, tc.seed = iidGraphRoot, seed
			checkRefused(t, tc, wantErr)
		})
	}
}

// adk-python passes yield_user_message to run_node_async for a Workflow root
// (runners.py), which yields the stored user event before the workflow's
// events; _run_with_trace, which runs other agent roots, ignores it.
func TestRun_GraphWorkflowRoot_YieldUserMessage(t *testing.T) {
	tests := []struct {
		name      string
		root      iidRootKind
		opts      []runner.RunOption
		wantFirst string
	}{
		{name: "graph_root_with_option", root: iidGraphRoot, opts: []runner.RunOption{runner.WithYieldUserMessage()}, wantFirst: "user text hi"},
		{name: "graph_root_without_option", root: iidGraphRoot, wantFirst: "root empty"},
		{name: "custom_agent_root_with_option", root: iidClassicRoot, opts: []runner.RunOption{runner.WithYieldUserMessage()}, wantFirst: "root text root reply"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := session.InMemoryService()
			newNodeTestSession(t, t.Context(), svc)
			root := tc.root(t)
			r := newIIDRunner(t, root, svc, false)

			events := runTurn(t, r, userText("hi"), tc.opts...)

			if len(events) == 0 {
				t.Fatal("Run() yielded no events")
			}
			if got := describeEvent(events[0]); got != tc.wantFirst {
				t.Errorf("first yielded event = %q, want %q (all: %q)", got, tc.wantFirst, describeEvents(events))
			}
			if stored := sessionEvents(t, svc); events[0].Author == "user" && events[0].ID != stored[0].ID {
				t.Errorf("yielded user event ID = %q, want the stored user event's %q", events[0].ID, stored[0].ID)
			}
			if got := root.runs.Load(); got != 1 {
				t.Errorf("root ran %d times, want 1", got)
			}
		})
	}
}

// A caller that stops ranging on the yielded user event stops the run before
// the workflow starts, and the after-run callbacks still run, as adk-python's
// finally covers the user-event yield (workflow/_node_runner_utils.py).
func TestRun_GraphWorkflowRoot_YieldUserMessage_CallerStopsOnUserEvent(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	root := iidGraphRoot(t)
	cb := &iidCallbacks{}
	r := newIIDRunner(t, root, svc, false, cb.plugin(t))

	var got []string
	for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}, runner.WithYieldUserMessage()) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		got = append(got, describeEvent(ev))
		break
	}

	if diff := cmp.Diff([]string{"user text hi"}, got); diff != "" {
		t.Errorf("yielded events (-want +got):\n%s", diff)
	}
	if n := root.runs.Load(); n != 0 {
		t.Errorf("root ran %d times, want 0", n)
	}
	if n := [2]int32{cb.beforeRun.Load(), cb.afterRun.Load()}; n != [2]int32{0, 1} {
		t.Errorf("callbacks [before run, after run] ran %v times, want [0 1]", n)
	}
}

// A retry stores no user event, so WithYieldUserMessage yields none: the
// first pair the run yields is the workflow's event.
func TestRun_GraphWorkflowRoot_YieldUserMessage_RetryYieldsNoUserEvent(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	root := iidGraphRoot(t)
	r := newIIDRunner(t, root, svc, false)
	runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-g"))

	var got []string
	for ev, err := range r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}, runner.WithInvocationID("inv-g"), runner.WithYieldUserMessage()) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev == nil {
			got = append(got, "nil event")
			continue
		}
		got = append(got, describeEvent(ev))
	}

	if diff := cmp.Diff([]string{"root empty"}, got); diff != "" {
		t.Errorf("yielded events (-want +got):\n%s", diff)
	}
}

// With no message, adk-python's node runtime stores a non-empty state delta
// in a user event without content and yields it with yield_user_message
// (workflow/_node_runner_utils.py).
func TestRun_GraphWorkflowRoot_YieldUserMessage_NilMessageYieldsDeltaEvent(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	root := iidGraphRoot(t)
	r := newIIDRunner(t, root, svc, false)

	events := runTurn(t, r, nil, runner.WithStateDelta(map[string]any{"k": "v"}), runner.WithYieldUserMessage())

	if diff := cmp.Diff([]string{"user empty", "root empty"}, describeEvents(events)); diff != "" {
		t.Fatalf("yielded events (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"k": "v"}, events[0].Actions.StateDelta); diff != "" {
		t.Errorf("events[0] state delta (-want +got):\n%s", diff)
	}
	if got := root.runs.Load(); got != 1 {
		t.Errorf("root ran %d times, want 1", got)
	}
}

// adk-python's node runtime stores a message with no parts as an empty user
// event and runs the workflow (workflow/_node_runner_utils.py); only the
// classic _append_new_message_to_session refuses it (runners.py).
func TestRun_GraphWorkflowRoot_EmptyMessageStored(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	root := iidGraphRoot(t)
	r := newIIDRunner(t, root, svc, false)

	runTurn(t, r, &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{}})

	stored := sessionEvents(t, svc)
	if len(stored) == 0 {
		t.Fatal("stored session has no events, want the empty user event first")
	}
	if first := stored[0]; first.Author != "user" || first.Content == nil || len(first.Content.Parts) != 0 {
		t.Errorf("stored event 0 = author %q content %+v, want an empty user event", first.Author, first.Content)
	}
	if got := root.runs.Load(); got != 1 {
		t.Errorf("root ran %d times, want 1", got)
	}
}

// On a retry (an invocation id whose invocation already stored a user
// message), adk-python's node runtime stores a non-empty state delta in a
// user event without content and yields it with yield_user_message
// (workflow/_node_runner_utils.py), on any runner.
func TestRun_GraphWorkflowRoot_RetryStateDeltaStored(t *testing.T) {
	for _, resumable := range []bool{false, true} {
		t.Run(fmt.Sprintf("resumable_%v", resumable), func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, seedUser("inv-0", "earlier"))
			root := iidGraphRoot(t)
			r := newIIDRunner(t, root, svc, resumable)

			runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-g"))
			events := runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-g"),
				runner.WithStateDelta(map[string]any{"k": "v2"}), runner.WithYieldUserMessage())

			if diff := cmp.Diff([]string{"user empty", "root empty"}, describeEvents(events)); diff != "" {
				t.Fatalf("yielded events (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(map[string]any{"k": "v2"}, events[0].Actions.StateDelta); diff != "" {
				t.Errorf("events[0] state delta (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]string{"user text hi", "user empty"}, storedUserEvents(t, svc, "inv-g")); diff != "" {
				t.Errorf("stored user events (-want +got):\n%s", diff)
			}
			if got, err := storedNodeTestSession(t, t.Context(), svc).State().Get("k"); err != nil || got != "v2" {
				t.Errorf("state[k] = %v, %v; want %q", got, err, "v2")
			}
		})
	}
}

// On a retry, or a run with no message, under an invocation id that already
// stored a user message, adk-python's node runtime runs the workflow on that
// stored message (workflow/_node_runner_utils.py, existing_user_content).
func TestRun_GraphWorkflowRoot_RetryRunsOnStoredMessage(t *testing.T) {
	tests := []struct {
		name  string
		retry *genai.Content
	}{
		{name: "different_text", retry: userText("second")},
		{name: "no_message", retry: nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := session.InMemoryService()
			newNodeTestSession(t, t.Context(), svc)
			var inputs []string
			r := newIIDRunner(t, iidRoot{agent: textRecordingGraph(t, &inputs)}, svc, false)

			runTurn(t, r, userText("first"), runner.WithInvocationID("inv-g"))
			runTurn(t, r, tc.retry, runner.WithInvocationID("inv-g"))

			if diff := cmp.Diff([]string{"first", "first"}, inputs); diff != "" {
				t.Errorf("workflow inputs (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]string{"user text first"}, storedUserEvents(t, svc, "inv-g")); diff != "" {
				t.Errorf("stored user events (-want +got):\n%s", diff)
			}
		})
	}
}

// On a resumable runner too, a root graph workflow's retry runs on the
// invocation's stored user message, as adk-python's node runtime does on any
// runner (workflow/_node_runner_utils.py, existing_user_content).
func TestRun_GraphWorkflowRoot_ResumableRetryRunsOnStoredMessage(t *testing.T) {
	svc := session.InMemoryService()
	seedResumeSession(t, svc, seedUser("inv-0", "earlier"))
	var inputs []string
	r := newIIDRunner(t, iidRoot{agent: textRecordingGraph(t, &inputs)}, svc, true)

	runTurn(t, r, userText("first"), runner.WithInvocationID("inv-g"))
	runTurn(t, r, userText("second"), runner.WithInvocationID("inv-g"))

	if diff := cmp.Diff([]string{"first", "first"}, inputs); diff != "" {
		t.Errorf("workflow inputs (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"user text first"}, storedUserEvents(t, svc, "inv-g")); diff != "" {
		t.Errorf("stored user events (-want +got):\n%s", diff)
	}
}

// An LlmAgent root runs on the node path, where a retry stores a non-empty
// state delta in a user event without content (workflow/_node_runner_utils.py).
func TestRun_LlmAgentRoot_RetryStateDeltaStored(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	r := newIIDRunner(t, iidLLMRoot(t), svc, false)

	runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-l"))
	runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-l"), runner.WithStateDelta(map[string]any{"k": "v2"}))

	if diff := cmp.Diff([]string{"user text hi", "user empty"}, storedUserEvents(t, svc, "inv-l")); diff != "" {
		t.Errorf("stored user events (-want +got):\n%s", diff)
	}
	if got, err := storedNodeTestSession(t, t.Context(), svc).State().Get("k"); err != nil || got != "v2" {
		t.Errorf("state[k] = %v, %v; want %q", got, err, "v2")
	}
}

// textRecordingGraph returns a root graph workflow "root" whose one step
// appends the text of the Content it receives to inputs, or "<none>" when it
// receives no Content.
func textRecordingGraph(t *testing.T, inputs *[]string) agent.Agent {
	t.Helper()
	step := workflow.NewFunctionNode("step", func(_ agent.Context, in *genai.Content) (string, error) {
		if in == nil {
			*inputs = append(*inputs, "<none>")
			return "", nil
		}
		*inputs = append(*inputs, contentText(in))
		return "", nil
	}, workflow.NodeConfig{})
	a, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, step)})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	return a
}
