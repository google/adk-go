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
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// userMessageCounter returns a plugin that counts its on-user-message
// callbacks and replaces a message with rewrite's result when rewrite is not
// nil and returns non-nil.
func userMessageCounter(t *testing.T, calls *atomic.Int32, rewrite func(*genai.Content) *genai.Content) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "count_user_message",
		OnUserMessageCallback: func(_ agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
			calls.Add(1)
			if rewrite == nil {
				return nil, nil
			}
			return rewrite(msg), nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

// replaceText returns a rewrite for userMessageCounter that replaces a
// message whose text is from with replacement, and leaves others unchanged.
func replaceText(from string, replacement *genai.Content) func(*genai.Content) *genai.Content {
	return func(msg *genai.Content) *genai.Content {
		if contentText(msg) == from {
			return replacement
		}
		return nil
	}
}

// contentText concatenates the text parts of c.
func contentText(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		if p != nil {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// storedUserEvents describes, with describeEvent, the stored user events of
// invocation inv.
func storedUserEvents(t *testing.T, svc session.Service, inv string) []string {
	t.Helper()
	var out []string
	for _, ev := range sessionEvents(t, svc) {
		if ev.Author == "user" && ev.InvocationID == inv {
			out = append(out, describeEvent(ev))
		}
	}
	return out
}

// recordingNode is a node root named "node" that counts its runs, records
// the user content and the input it sees and the invocation it runs in, and
// yields one event with Output "node-out".
type recordingNode struct {
	runs         atomic.Int32
	mu           sync.Mutex
	userContents []string
	inputs       []string
	invocations  []string
}

func (r *recordingNode) node() workflow.Node {
	return newFuncNode("node", func(ctx agent.Context, input any, yield func(*session.Event, error) bool) {
		r.runs.Add(1)
		in, _ := input.(*genai.Content)
		r.mu.Lock()
		r.userContents = append(r.userContents, contentText(ctx.UserContent()))
		r.inputs = append(r.inputs, contentText(in))
		r.invocations = append(r.invocations, ctx.InvocationID())
		r.mu.Unlock()
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = "node-out"
		yield(ev, nil)
	})
}

// retryRunner returns a non-resumable runner over the custom root "root"
// that rec records, or, when useNode is true, over the node root that node
// records.
func retryRunner(t *testing.T, svc session.Service, useNode bool, rec *recordingRoot, node *recordingNode, plugins ...*plugin.Plugin) *runner.Runner {
	t.Helper()
	if useNode {
		return newNodeRootRunner(t, node.node(), svc, plugins...)
	}
	return newResumeRunner(t, rec.agent(t, "root"), svc, false, plugins...)
}

func TestRun_WithInvocationID_BareNodeRoot(t *testing.T) {
	t.Run("custom_id", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		var rec recordingNode
		r := newNodeRootRunner(t, rec.node(), svc)

		events := runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-node-1"))

		if len(events) != 1 || events[0].InvocationID != "inv-node-1" || events[0].Output != "node-out" {
			t.Fatalf("Run() yielded %d events, first = %+v; want one event with invocation %q and output %q", len(events), firstOrNil(events), "inv-node-1", "node-out")
		}
		stored := sessionEvents(t, svc)
		if diff := cmp.Diff([]string{"inv-node-1", "inv-node-1"}, invocationIDs(stored)); diff != "" {
			t.Errorf("stored invocation IDs (-want +got):\n%s", diff)
		}
		if stored[0].Author != "user" {
			t.Errorf("stored[0].Author = %q, want %q", stored[0].Author, "user")
		}
	})

	t.Run("empty_id_mints_uuid", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		var rec recordingNode
		r := newNodeRootRunner(t, rec.node(), svc)

		events := runTurn(t, r, userText("hi"), runner.WithInvocationID(""))

		if len(events) != 1 {
			t.Fatalf("Run() yielded %d events, want 1", len(events))
		}
		if got := events[0].InvocationID; !strings.HasPrefix(got, "e-") || len(got) != 38 {
			t.Errorf("events[0].InvocationID = %q, want \"e-\" and a 36-character UUID", got)
		}
	})
}

func TestRun_RetryDeduplication_EdgeCases(t *testing.T) {
	image := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0x89, 0x50}}},
	}}
	rewind := resumeSeed("inv-rewind", "user", nil)
	rewind.Actions.RewindBeforeInvocationID = "inv-rw"
	deltaOnly := resumeSeed("inv-sd", "user", nil)
	deltaOnly.Actions.StateDelta = map[string]any{"k": "v"}

	tests := []struct {
		name      string
		node      bool
		resumable bool
		seed      []*session.Event
		// first, when not nil, is sent under inv before retry.
		first, retry *genai.Content
		inv          string
		wantCalls    int32
		wantUser     []string
		wantRootRuns int32
	}{
		{
			// A retry is recognized by the ID alone (adk-python runners.py
			// _handle_new_message), so different text is dropped.
			name:      "different_text_classic",
			first:     userText("book a flight"),
			retry:     userText("cancel it"),
			inv:       "inv-diff",
			wantCalls: 1, wantUser: []string{"user text book a flight"}, wantRootRuns: 2,
		},
		{
			name:      "different_text_node",
			node:      true,
			first:     userText("book a flight"),
			retry:     userText("cancel it"),
			inv:       "inv-diff",
			wantCalls: 1, wantUser: []string{"user text book a flight"}, wantRootRuns: 2,
		},
		{
			// A resumable runner resumes the invocation and runs the root
			// again, and the retry stores no second user event.
			name:      "same_text_resumable_classic",
			resumable: true,
			seed:      []*session.Event{seedUser("inv-same", "hello")},
			retry:     userText("hello"),
			inv:       "inv-same",
			wantCalls: 0, wantUser: []string{"user text hello"}, wantRootRuns: 1,
		},
		{
			name:      "image_only_first_turn",
			first:     image,
			retry:     userText("follow up"),
			inv:       "inv-img",
			wantCalls: 1, wantUser: []string{"user text "}, wantRootRuns: 2,
		},
		{
			name:      "rewound_invocation_not_deduplicated",
			seed:      []*session.Event{seedUser("inv-rw", "old"), seedText("inv-rw", "root", "root reply"), rewind},
			retry:     userText("fresh text"),
			inv:       "inv-rw",
			wantCalls: 1, wantUser: []string{"user text old", "user text fresh text"}, wantRootRuns: 1,
		},
		{
			name:      "only_function_response_in_invocation_not_deduplicated",
			seed:      []*session.Event{resumeSeed("inv-fr-only", "user", responseContent("fc-0", "tool", map[string]any{}))},
			retry:     userText("first text"),
			inv:       "inv-fr-only",
			wantCalls: 1, wantUser: []string{"user response tool", "user text first text"}, wantRootRuns: 1,
		},
		{
			name:      "only_state_delta_in_invocation_not_deduplicated",
			seed:      []*session.Event{deltaOnly},
			retry:     userText("first text"),
			inv:       "inv-sd",
			wantCalls: 1, wantUser: []string{"user empty", "user text first text"}, wantRootRuns: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, tc.seed...)
			var calls atomic.Int32
			var rec recordingRoot
			var node recordingNode
			counter := userMessageCounter(t, &calls, nil)
			r := retryRunner(t, svc, tc.node, &rec, &node, counter)
			if tc.resumable {
				r = newResumeRunner(t, rec.agent(t, "root"), svc, true, counter)
			}

			if tc.first != nil {
				runTurn(t, r, tc.first, runner.WithInvocationID(tc.inv))
			}
			runTurn(t, r, tc.retry, runner.WithInvocationID(tc.inv))

			if got := calls.Load(); got != tc.wantCalls {
				t.Errorf("OnUserMessageCallback ran %d times, want %d", got, tc.wantCalls)
			}
			if diff := cmp.Diff(tc.wantUser, storedUserEvents(t, svc, tc.inv)); diff != "" {
				t.Errorf("stored user events of %s (-want +got):\n%s", tc.inv, diff)
			}
			runs, invocations := rec.runs.Load(), rec.invocations
			if tc.node {
				runs, invocations = node.runs.Load(), node.invocations
			}
			if runs != tc.wantRootRuns {
				t.Errorf("root ran %d times, want %d", runs, tc.wantRootRuns)
			}
			for i, got := range invocations {
				if got != tc.inv {
					t.Errorf("root run %d invocation = %q, want %q", i, got, tc.inv)
				}
			}
		})
	}
}

func TestRun_RetryUserContentByPath(t *testing.T) {
	// The plugin trims every message it sees, so a callback that ran on the
	// retry would make the node see "retry text".
	rewrite := func(msg *genai.Content) *genai.Content { return userText(strings.TrimSpace(contentText(msg))) }

	t.Run("classic_path", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		var calls atomic.Int32
		var rec recordingRoot
		r := newResumeRunner(t, rec.agent(t, "root"), svc, false, userMessageCounter(t, &calls, rewrite))

		runTurn(t, r, userText("  clean  "), runner.WithInvocationID("inv-b3"))
		runTurn(t, r, userText("retry text"), runner.WithInvocationID("inv-b3"))

		if diff := cmp.Diff([]string{"clean", "retry text"}, rec.userContents); diff != "" {
			t.Errorf("root user contents (-want +got):\n%s", diff)
		}
	})

	t.Run("node_path_bare_node", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		var calls atomic.Int32
		var node recordingNode
		r := newNodeRootRunner(t, node.node(), svc, userMessageCounter(t, &calls, rewrite))

		runTurn(t, r, userText("  clean  "), runner.WithInvocationID("inv-b3"))
		runTurn(t, r, userText("retry text"), runner.WithInvocationID("inv-b3"))

		if diff := cmp.Diff([]string{"clean", "clean"}, node.userContents); diff != "" {
			t.Errorf("node user contents (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"clean", "clean"}, node.inputs); diff != "" {
			t.Errorf("node inputs (-want +got):\n%s", diff)
		}
	})

	t.Run("node_path_llm_agent", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		var calls atomic.Int32
		var seen []string
		beforeRun, err := plugin.New(plugin.Config{
			Name: "record_user_content",
			BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
				seen = append(seen, contentText(ctx.UserContent()))
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("plugin.New() error = %v", err)
		}
		r := newResumeRunner(t, textReplyLlmAgent(t, "root", "reply"), svc, false, userMessageCounter(t, &calls, rewrite), beforeRun)

		runTurn(t, r, userText("  clean  "), runner.WithInvocationID("inv-b3"))
		runTurn(t, r, userText("retry text"), runner.WithInvocationID("inv-b3"))

		if diff := cmp.Diff([]string{"clean", "clean"}, seen); diff != "" {
			t.Errorf("BeforeRunCallback user contents (-want +got):\n%s", diff)
		}
	})
}

func TestRun_RetryStateDeltaDroppedOnClassicPath(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	var rec recordingRoot
	r := newResumeRunner(t, rec.agent(t, "root"), svc, false)

	runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-b4-c"), runner.WithStateDelta(map[string]any{"k": "v1"}))
	runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-b4-c"), runner.WithStateDelta(map[string]any{"k": "v2", "retry_key": "retry_val"}))

	if diff := cmp.Diff([]string{"user text hi"}, storedUserEvents(t, svc, "inv-b4-c")); diff != "" {
		t.Errorf("stored user events (-want +got):\n%s", diff)
	}
	state := storedNodeTestSession(t, t.Context(), svc).State()
	if got, err := state.Get("k"); err != nil || got != "v1" {
		t.Errorf("state[k] = %v, %v; want %q", got, err, "v1")
	}
	if got, err := state.Get("retry_key"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("state[retry_key] = %v, %v; want error %v", got, err, session.ErrStateKeyNotExist)
	}
	if got := rec.runs.Load(); got != 2 {
		t.Errorf("root ran %d times, want 2", got)
	}
}

func TestRun_RetryStateDeltaAppendedOnNodePath(t *testing.T) {
	t.Run("non_empty_delta_yielded", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		var node recordingNode
		r := newNodeRootRunner(t, node.node(), svc)

		runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-b4-n"))
		events := runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-b4-n"),
			runner.WithStateDelta(map[string]any{"k": "v2"}), runner.WithYieldUserMessage())

		if len(events) == 0 {
			t.Fatal("Run() yielded no events, want the delta-only user event first")
		}
		first := events[0]
		if first.Author != "user" || first.InvocationID != "inv-b4-n" || first.Content != nil {
			t.Errorf("events[0] = author %q invocation %q content %v, want author %q invocation %q content nil", first.Author, first.InvocationID, first.Content, "user", "inv-b4-n")
		}
		if diff := cmp.Diff(map[string]any{"k": "v2"}, first.Actions.StateDelta); diff != "" {
			t.Errorf("events[0] state delta (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]string{"user text hi", "user empty"}, storedUserEvents(t, svc, "inv-b4-n")); diff != "" {
			t.Errorf("stored user events (-want +got):\n%s", diff)
		}
		if got, err := storedNodeTestSession(t, t.Context(), svc).State().Get("k"); err != nil || got != "v2" {
			t.Errorf("state[k] = %v, %v; want %q", got, err, "v2")
		}
	})

	t.Run("empty_delta_appends_nothing", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		var node recordingNode
		r := newNodeRootRunner(t, node.node(), svc)

		runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-b4-n"))
		runTurn(t, r, userText("hi"), runner.WithInvocationID("inv-b4-n"))

		if diff := cmp.Diff([]string{"user text hi"}, storedUserEvents(t, svc, "inv-b4-n")); diff != "" {
			t.Errorf("stored user events (-want +got):\n%s", diff)
		}
	})
}

func TestRun_FunctionResponseOverridesCallerInvocationID(t *testing.T) {
	rewindOld := resumeSeed("inv_rw", "user", nil)
	rewindOld.Actions.RewindBeforeInvocationID = "inv_old"
	answer := responseContent("fc-1", "tool", map[string]any{"ok": true})
	answerBoth := &genai.Content{Role: genai.RoleUser, Parts: append(
		responseContent("fc-1", "tool", map[string]any{}).Parts,
		responseContent("fc-2", "tool", map[string]any{}).Parts...)}

	type root int
	const (
		classic root = iota
		classicResumable
		nodeRoot
		llmRoot
	)
	tests := []struct {
		name    string
		root    root
		seed    []*session.Event
		msg     *genai.Content
		wantInv string
	}{
		{
			name:    "classic_non_resumable",
			root:    classic,
			seed:    []*session.Event{seedUser("inv_call", "start"), resumeSeed("inv_call", "root", fnCallContent("fc-1", "tool", nil))},
			msg:     answer,
			wantInv: "inv_call",
		},
		{
			name:    "classic_resumable",
			root:    classicResumable,
			seed:    []*session.Event{seedUser("inv_call", "start"), resumeSeed("inv_call", "root", fnCallContent("fc-1", "tool", nil))},
			msg:     answer,
			wantInv: "inv_call",
		},
		{
			name:    "node_root",
			root:    nodeRoot,
			seed:    []*session.Event{seedUser("inv_call", "start"), resumeSeed("inv_call", "node", fnCallContent("fc-1", "tool", nil))},
			msg:     answer,
			wantInv: "inv_call",
		},
		{
			name:    "llm_agent_root",
			root:    llmRoot,
			seed:    []*session.Event{seedUser("inv_call", "start"), resumeSeed("inv_call", "root", fnCallContent("fc-1", "tool", nil))},
			msg:     answer,
			wantInv: "inv_call",
		},
		{
			name: "two_call_events_same_invocation",
			root: classic,
			seed: []*session.Event{
				seedUser("inv_call", "start"),
				resumeSeed("inv_call", "root", fnCallContent("fc-1", "tool", nil)),
				resumeSeed("inv_call", "root", fnCallContent("fc-2", "tool", nil)),
			},
			msg:     answerBoth,
			wantInv: "inv_call",
		},
		{
			// The newest call with the ID was rewound, so the live one wins.
			name: "rewound_call_event_ignored",
			root: classic,
			seed: []*session.Event{
				seedUser("inv_live", "start"),
				resumeSeed("inv_live", "root", fnCallContent("fc-1", "tool", nil)),
				seedUser("inv_old", "later"),
				resumeSeed("inv_old", "root", fnCallContent("fc-1", "tool", nil)),
				rewindOld,
			},
			msg:     answer,
			wantInv: "inv_live",
		},
		{
			// The ID was called in two invocations; adk-python scans the
			// history newest first (runners.py:679), so the newer one wins.
			name: "same_call_id_in_two_invocations_newest_wins",
			root: classic,
			seed: []*session.Event{
				seedUser("inv_1", "first"),
				resumeSeed("inv_1", "root", fnCallContent("fc-1", "tool", nil)),
				resumeSeed("inv_1", "user", responseContent("fc-1", "tool", map[string]any{})),
				seedUser("inv_2", "second"),
				resumeSeed("inv_2", "root", fnCallContent("fc-1", "tool", nil)),
			},
			msg:     answer,
			wantInv: "inv_2",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, tc.seed...)
			var rec recordingRoot
			var node recordingNode
			var r *runner.Runner
			switch tc.root {
			case classic:
				r = newResumeRunner(t, rec.agent(t, "root"), svc, false)
			case classicResumable:
				r = newResumeRunner(t, rec.agent(t, "root"), svc, true)
			case nodeRoot:
				r = newNodeRootRunner(t, node.node(), svc)
			case llmRoot:
				r = newResumeRunner(t, textReplyLlmAgent(t, "root", "reply"), svc, false)
			}

			events := runTurn(t, r, tc.msg, runner.WithInvocationID("caller-override"))

			if len(events) == 0 {
				t.Fatal("Run() yielded no events")
			}
			for i, ev := range events {
				if ev.InvocationID != tc.wantInv {
					t.Errorf("events[%d].InvocationID = %q, want %q", i, ev.InvocationID, tc.wantInv)
				}
			}
			stored := sessionEvents(t, svc)
			userFR := stored[len(tc.seed)]
			if userFR.Author != "user" || userFR.InvocationID != tc.wantInv {
				t.Errorf("stored[%d] = author %q invocation %q, want the user's response under %q", len(tc.seed), userFR.Author, userFR.InvocationID, tc.wantInv)
			}
			for _, ev := range stored {
				if ev.InvocationID == "caller-override" {
					t.Errorf("stored event %q carries the caller's invocation ID", describeEvent(ev))
				}
			}
		})
	}
}

func TestRun_LenientResolution_PluginReplacementMultiInvocation(t *testing.T) {
	answerTwo := &genai.Content{Role: genai.RoleUser, Parts: append(
		responseContent("fc-a", "tool", map[string]any{}).Parts,
		responseContent("fc-b", "tool", map[string]any{}).Parts...)}

	// adk-python's classic path never re-resolves a replacement, so a classic
	// root keeps its minted ID even on a resumable runner.
	for _, resumable := range []bool{false, true} {
		name := "classic_root_keeps_minted_id"
		if resumable {
			name = "resumable_classic_root_keeps_minted_id"
		}
		t.Run(name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc,
				resumeSeed("inv_a", "root", fnCallContent("fc-a", "tool", nil)),
				resumeSeed("inv_b", "root", fnCallContent("fc-b", "tool", nil)))
			var calls atomic.Int32
			var rec recordingRoot
			r := newResumeRunner(t, rec.agent(t, "root"), svc, resumable, userMessageCounter(t, &calls, replaceText("hi", answerTwo)))

			events := runTurn(t, r, userText("hi"))

			if len(events) != 1 {
				t.Fatalf("Run() yielded %d events, want 1", len(events))
			}
			if got := events[0].InvocationID; got == "inv_a" || got == "inv_b" || !strings.HasPrefix(got, "e-") {
				t.Errorf("events[0].InvocationID = %q, want a minted \"e-\" ID", got)
			}
		})
	}

	// A root adk-python runs as a node re-resolves a replacement strictly
	// (workflow/_node_runner_utils.py, run_node_async).
	t.Run("node_root_refuses", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc,
			resumeSeed("inv_a", "node", fnCallContent("fc-a", "tool", nil)),
			resumeSeed("inv_b", "node", fnCallContent("fc-b", "tool", nil)))
		var calls atomic.Int32
		var node recordingNode
		r := newNodeRootRunner(t, node.node(), svc, userMessageCounter(t, &calls, replaceText("hi", answerTwo)))

		events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}))

		const wantErr = "function responses resolve to multiple invocations (2): all function responses in a single message must belong to the same invocation"
		if len(errs) != 1 || errs[0].Error() != wantErr {
			t.Errorf("Run() errors = %v, want [%s]", errs, wantErr)
		}
		if len(events) != 0 || node.runs.Load() != 0 {
			t.Errorf("Run() yielded %d events and ran the node %d times, want 0 and 0", len(events), node.runs.Load())
		}
		if got := len(sessionEvents(t, svc)); got != 2 {
			t.Errorf("session holds %d events, want the 2 seeded ones", got)
		}
	})

	// The node-path re-resolution of a replacement has the run's mode. On a
	// resumable runner it is strict, so a replacement answering an unknown
	// call is refused before anything is stored.
	t.Run("resumable_node_root_refuses_unmatched", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc, seedUser("inv_a", "first"), resumeSeed("inv_a", "root", fnCallContent("fc-a", "tool", nil)))
		var calls atomic.Int32
		root := iidNodeRoot(t)
		r := newIIDRunner(t, root, svc, true, userMessageCounter(t, &calls, replaceText("hi", iidResponses("fc-missing"))))

		events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}))

		const wantErr = "function call not found for 1 function response ids: each function response ID must match a function call in the session history"
		if len(errs) != 1 || errs[0].Error() != wantErr {
			t.Errorf("Run() errors = %v, want [%s]", errs, wantErr)
		}
		if len(events) != 0 || root.runs.Load() != 0 {
			t.Errorf("Run() yielded %d events and ran the node %d times, want 0 and 0", len(events), root.runs.Load())
		}
		if got := len(sessionEvents(t, svc)); got != 2 {
			t.Errorf("session holds %d events, want the 2 seeded ones", got)
		}
	})

	// On a runner that is not resumable, with no caller ID, the same
	// replacement starts a new invocation, where adk-python refuses it (see
	// resolutionMode).
	t.Run("non_resumable_node_root_starts_new_invocation_on_unmatched", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc, seedUser("inv_a", "first"), resumeSeed("inv_a", "root", fnCallContent("fc-a", "tool", nil)))
		var calls atomic.Int32
		root := iidNodeRoot(t)
		r := newIIDRunner(t, root, svc, false, userMessageCounter(t, &calls, replaceText("hi", iidResponses("fc-missing"))))

		events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, userText("hi"), agent.RunConfig{}))

		if len(errs) != 0 || root.runs.Load() != 1 {
			t.Fatalf("Run() errors = %v and ran the node %d times, want none and 1", errs, root.runs.Load())
		}
		if len(events) != 1 || !strings.HasPrefix(events[0].InvocationID, "e-") {
			t.Errorf("Run() yielded %v under %v, want one event under a minted \"e-\" ID", describeEvents(events), invocationIDs(events))
		}
	})

	// adk-python re-resolves a replacement only when the run has no
	// invocation ID (workflow/_node_runner_utils.py: `if not invocation_id:`).
	for _, useNode := range []bool{false, true} {
		name := "caller_id_survives_replacement_classic"
		author := "root"
		if useNode {
			name, author = "caller_id_survives_replacement_node", "node"
		}
		t.Run(name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, seedUser("inv-call", "start"), resumeSeed("inv-call", author, fnCallContent("fc-1", "tool", nil)))
			var calls atomic.Int32
			var rec recordingRoot
			var node recordingNode
			plug := userMessageCounter(t, &calls, replaceText("hi", responseContent("fc-1", "tool", map[string]any{})))
			r := retryRunner(t, svc, useNode, &rec, &node, plug)

			events := runTurn(t, r, userText("hi"), runner.WithInvocationID("caller-inv"))

			if diff := cmp.Diff([]string{"caller-inv"}, invocationIDs(events)); diff != "" {
				t.Errorf("yielded invocation IDs (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]string{"user response tool"}, storedUserEvents(t, svc, "caller-inv")); diff != "" {
				t.Errorf("stored user events of caller-inv (-want +got):\n%s", diff)
			}
		})
	}
}

// On the node path a replacement that answers a call stays the turn's
// message, which resumes the paused node, while the node reads the
// invocation's stored user message (adk-python workflow/_node_runner_utils.py,
// run_node_async).
func TestRun_NodeReplacementAnsweringCallKeepsStoredUserContent(t *testing.T) {
	svc := session.InMemoryService()
	newNodeTestSession(t, t.Context(), svc)
	var mu sync.Mutex
	var userContents []string
	n := newFuncNode("node", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		mu.Lock()
		userContents = append(userContents, contentText(ctx.UserContent()))
		mu.Unlock()
		ev := session.NewEvent(ctx, ctx.InvocationID())
		if _, ok := ctx.ResumedInput("fc-1"); ok {
			ev.Output = "resumed"
			yield(ev, nil)
			return
		}
		ev.Content = fnCallContent("fc-1", "tool", map[string]any{})
		ev.LongRunningToolIDs = []string{"fc-1"}
		yield(ev, nil)
	})
	var calls atomic.Int32
	plug := userMessageCounter(t, &calls, replaceText("approve", responseContent("fc-1", "tool", map[string]any{"v": 1})))
	r := newNodeRootRunner(t, n, svc, plug)

	turn1 := runTurn(t, r, userText("original text"))
	turn2 := runTurn(t, r, userText("approve"))

	if diff := cmp.Diff([]string{"original text", "original text"}, userContents); diff != "" {
		t.Errorf("node user contents (-want +got):\n%s", diff)
	}
	var outputs []any
	for _, ev := range turn2 {
		if ev.InvocationID != turn1[0].InvocationID {
			t.Errorf("turn 2 event %q invocation = %q, want turn 1's %q", describeEvent(ev), ev.InvocationID, turn1[0].InvocationID)
		}
		if ev.Output != nil {
			outputs = append(outputs, ev.Output)
		}
	}
	if diff := cmp.Diff([]any{"resumed"}, outputs); diff != "" {
		t.Errorf("turn 2 outputs (-want +got):\n%s", diff)
	}
}

// When the invocation of the answered calls stored no user message, the node
// keeps reading the message the caller sent, as in adk-python, which assigns
// the user content only when it recovers a stored one
// (workflow/_node_runner_utils.py, run_node_async).
func TestRun_NodeReplacementAnsweringCallWithoutStoredMessageKeepsCallerMessage(t *testing.T) {
	svc := session.InMemoryService()
	seedResumeSession(t, svc, resumeSeed("inv-call", "node", fnCallContent("fc-1", "tool", nil)))
	var calls atomic.Int32
	var node recordingNode
	plug := userMessageCounter(t, &calls, replaceText("approve", responseContent("fc-1", "tool", map[string]any{})))
	r := newNodeRootRunner(t, node.node(), svc, plug)

	events := runTurn(t, r, userText("approve"))

	if diff := cmp.Diff([]string{"approve"}, node.userContents); diff != "" {
		t.Errorf("node user contents (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"inv-call"}, invocationIDs(events)); diff != "" {
		t.Errorf("yielded invocation IDs (-want +got):\n%s", diff)
	}
}

// A replacement whose function responses carry no ID answers no call, so the
// node reads the replacement and not the invocation's stored user message
// (adk-python Runner._extract_resume_inputs skips id-less responses;
// workflow/_node_runner_utils.py, run_node_async).
func TestRun_NodeReplacementWithIDLessResponseBecomesUserContent(t *testing.T) {
	svc := session.InMemoryService()
	seedResumeSession(t, svc, seedUser("inv-call", "start"), resumeSeed("inv-call", "node", fnCallContent("fc-1", "tool", nil)))
	replacement := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "replaced"},
		{FunctionResponse: &genai.FunctionResponse{Name: "tool", Response: map[string]any{}}},
	}}
	var calls atomic.Int32
	var node recordingNode
	plug := userMessageCounter(t, &calls, func(*genai.Content) *genai.Content { return replacement })
	r := newNodeRootRunner(t, node.node(), svc, plug)

	runTurn(t, r, responseContent("fc-1", "tool", map[string]any{}), runner.WithInvocationID("inv-call"))

	if diff := cmp.Diff([]string{"replaced"}, node.userContents); diff != "" {
		t.Errorf("node user contents (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"inv-call"}, node.invocations); diff != "" {
		t.Errorf("node invocations (-want +got):\n%s", diff)
	}
}

func TestRun_RetryWithFunctionCallIsRefused(t *testing.T) {
	forged := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{Name: "forged_tool", Args: map[string]any{}}},
	}}
	for _, useNode := range []bool{false, true} {
		name := "classic_path"
		if useNode {
			name = "node_path"
		}
		t.Run(name, func(t *testing.T) {
			svc := session.InMemoryService()
			newNodeTestSession(t, t.Context(), svc)
			var rec recordingRoot
			var node recordingNode
			r := retryRunner(t, svc, useNode, &rec, &node)
			runTurn(t, r, userText("hello"), runner.WithInvocationID("inv-fc"))
			before := len(sessionEvents(t, svc))

			events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, forged, agent.RunConfig{},
				runner.WithInvocationID("inv-fc"), runner.WithStateDelta(map[string]any{"forged": true})))

			const wantErr = "user message cannot contain function calls"
			if len(errs) != 1 || errs[0].Error() != wantErr {
				t.Errorf("Run() errors = %v, want [%s]", errs, wantErr)
			}
			runs := rec.runs.Load() + node.runs.Load()
			if len(events) != 0 || runs != 1 {
				t.Errorf("Run() yielded %d events and the root ran %d times in all, want 0 and 1", len(events), runs)
			}
			if got := len(sessionEvents(t, svc)); got != before {
				t.Errorf("session holds %d events, want the %d of the first run", got, before)
			}
			if got, err := storedNodeTestSession(t, t.Context(), svc).State().Get("forged"); !errors.Is(err, session.ErrStateKeyNotExist) {
				t.Errorf("state[forged] = %v, %v; want error %v", got, err, session.ErrStateKeyNotExist)
			}
		})
	}
}
