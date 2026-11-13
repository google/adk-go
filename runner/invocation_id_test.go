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
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// iidRoot is the root of a runner under test: exactly one of agent and
// node is set. runs counts how often the root did its work.
type iidRoot struct {
	agent agent.Agent
	node  workflow.Node
	runs  *atomic.Int32
}

// iidRootKind builds a root named "root" that yields one event per run.
type iidRootKind func(t *testing.T) iidRoot

// iidClassicRoot is a custom agent, which runs on the classic path.
func iidClassicRoot(t *testing.T) iidRoot {
	t.Helper()
	runs := &atomic.Int32{}
	a, err := agent.New(agent.Config{
		Name: "root",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				runs.Add(1)
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Content = modelText("root reply")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return iidRoot{agent: a, runs: runs}
}

// iidLLMRoot is an LLM agent, which runs on the node path.
func iidLLMRoot(t *testing.T) iidRoot {
	t.Helper()
	m := &iidCountingModel{runs: &atomic.Int32{}}
	return iidRoot{agent: scriptedLlmAgent(t, "root", m, llmagent.Config{}), runs: m.runs}
}

// iidNodeRoot is a Config.Node root.
func iidNodeRoot(*testing.T) iidRoot {
	runs := &atomic.Int32{}
	n := newFuncNode("root", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		runs.Add(1)
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = "node-out"
		yield(ev, nil)
	})
	return iidRoot{node: n, runs: runs}
}

// iidGraphRoot is a graph workflow agent built by workflowagent.New.
func iidGraphRoot(t *testing.T) iidRoot {
	t.Helper()
	runs := &atomic.Int32{}
	step := workflow.NewFunctionNode("step", func(agent.Context, any) (string, error) {
		runs.Add(1)
		return "step-out", nil
	}, workflow.NodeConfig{})
	a, err := workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, step)})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	return iidRoot{agent: a, runs: runs}
}

// iidCountingModel answers every request with the text "llm reply".
type iidCountingModel struct {
	runs *atomic.Int32
}

func (m *iidCountingModel) Name() string { return "counting" }

func (m *iidCountingModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.runs.Add(1)
		yield(&model.LLMResponse{Content: modelText("llm reply")}, nil)
	}
}

// iidCallbacks counts the run-level plugin callbacks.
type iidCallbacks struct {
	onUserMessage, beforeRun, afterRun atomic.Int32
}

func (c *iidCallbacks) plugin(t *testing.T) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "count_run_callbacks",
		OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
			c.onUserMessage.Add(1)
			return nil, nil
		},
		BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
			c.beforeRun.Add(1)
			return nil, nil
		},
		AfterRunCallback: func(agent.InvocationContext) {
			c.afterRun.Add(1)
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

func newIIDRunner(t *testing.T, root iidRoot, svc session.Service, resumable bool, plugins ...*plugin.Plugin) *runner.Runner {
	t.Helper()
	r, err := runner.New(runner.Config{
		AppName:        nodeTestApp,
		Agent:          root.agent,
		Node:           root.node,
		SessionService: svc,
		Resumable:      resumable,
		PluginConfig:   runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r
}

// iidResponses returns a user message answering the calls ids, in order.
func iidResponses(ids ...string) *genai.Content {
	msg := &genai.Content{Role: genai.RoleUser}
	for _, id := range ids {
		msg.Parts = append(msg.Parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "tool", Response: map[string]any{"ok": true}}})
	}
	return msg
}

// iidTwoInvocations seeds a call to fc-a in inv_a and a call to fc-b in
// inv_b, each after its invocation's user message.
func iidTwoInvocations() []*session.Event {
	return []*session.Event{
		seedUser("inv_a", "first"),
		resumeSeed("inv_a", "root", fnCallContent("fc-a", "tool", nil)),
		seedUser("inv_b", "second"),
		resumeSeed("inv_b", "root", fnCallContent("fc-b", "tool", nil)),
	}
}

func errorTexts(errs []error) []string {
	var out []string
	for _, err := range errs {
		out = append(out, err.Error())
	}
	return out
}

// iidRefusalCase is a run that must be refused before it changes anything.
type iidRefusalCase struct {
	name      string
	root      iidRootKind
	resumable bool
	seed      []*session.Event
	msg       *genai.Content
	opts      []runner.RunOption
}

// checkRefused runs tc and checks that Run yields only wantErr, stores no
// event, runs no run-level callback and does not run the root.
func checkRefused(t *testing.T, tc iidRefusalCase, wantErr string) {
	t.Helper()
	svc := session.InMemoryService()
	seedResumeSession(t, svc, tc.seed...)
	root := tc.root(t)
	cb := &iidCallbacks{}
	r := newIIDRunner(t, root, svc, tc.resumable, cb.plugin(t))

	events, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, tc.msg, agent.RunConfig{}, tc.opts...))

	if len(errs) != 1 || errs[0].Error() != wantErr {
		t.Errorf("Run() errors = %q, want [%q]", errorTexts(errs), wantErr)
	}
	if len(events) != 0 {
		t.Errorf("Run() yielded %v, want nothing", describeEvents(events))
	}
	if got := len(sessionEvents(t, svc)); got != len(tc.seed) {
		t.Errorf("session holds %d events, want the %d seeded", got, len(tc.seed))
	}
	if got := root.runs.Load(); got != 0 {
		t.Errorf("root ran %d times, want 0", got)
	}
	if got := [3]int32{cb.onUserMessage.Load(), cb.beforeRun.Load(), cb.afterRun.Load()}; got != [3]int32{} {
		t.Errorf("callbacks [on user message, before run, after run] ran %v times, want [0 0 0]", got)
	}
}

func TestRun_StrictResolution_UnmatchedFunctionResponse(t *testing.T) {
	const wantErr = "function call not found for 2 function response ids: each function response ID must match a function call in the session history"
	unmatched := iidResponses("fc-missing-1", "fc-missing-2")
	seed := []*session.Event{
		seedUser("inv_a", "first"),
		resumeSeed("inv_a", "root", fnCallContent("fc-a", "tool", nil)),
	}
	tests := []iidRefusalCase{
		{name: "resumable_classic_root", root: iidClassicRoot, resumable: true},
		{name: "node_root_resumable", root: iidNodeRoot, resumable: true},
		{name: "node_root_with_caller_id", root: iidNodeRoot, opts: []runner.RunOption{runner.WithInvocationID("x")}},
		{name: "graph_workflow_root_resumable", root: iidGraphRoot, resumable: true},
		{name: "graph_workflow_root_with_caller_id", root: iidGraphRoot, opts: []runner.RunOption{runner.WithInvocationID("x")}},
		{name: "llm_agent_root_resumable", root: iidLLMRoot, resumable: true},
		{name: "llm_agent_root_with_caller_id", root: iidLLMRoot, opts: []runner.RunOption{runner.WithInvocationID("x")}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.seed, tc.msg = seed, unmatched
			checkRefused(t, tc, wantErr)
		})
	}
}

// A message mixing text with a function response is refused as mixed before
// its invocation is resolved, even when strict resolution would refuse the
// unknown call (adk-python Runner.run_node_async).
func TestRun_StrictResolution_MixedMessageRefusedBeforeResolution(t *testing.T) {
	const wantErr = "message cannot contain both function responses and text: function responses resume an existing invocation while text starts a new one"
	mixed := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "hi"},
		{FunctionResponse: &genai.FunctionResponse{ID: "fc-unknown", Name: "tool", Response: map[string]any{"ok": true}}},
	}}
	withCaller := []runner.RunOption{runner.WithInvocationID("x")}
	tests := []iidRefusalCase{
		{name: "node_root_with_caller_id", root: iidNodeRoot, opts: withCaller},
		{name: "node_root_resumable", root: iidNodeRoot, resumable: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.seed, tc.msg = []*session.Event{seedUser("inv_a", "first")}, mixed
			checkRefused(t, tc, wantErr)
		})
	}
}

func TestRun_StrictResolution_MultiInvocationOnNodeAndGraphRoots(t *testing.T) {
	const wantErr = "function responses resolve to multiple invocations (2): all function responses in a single message must belong to the same invocation"
	tests := []iidRefusalCase{
		{name: "node_root", root: iidNodeRoot},
		{name: "graph_workflow_root", root: iidGraphRoot},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.seed, tc.msg = iidTwoInvocations(), iidResponses("fc-a", "fc-b")
			checkRefused(t, tc, wantErr)
		})
	}
}

func TestRun_StrictResolution_IDLessFirstFunctionResponse(t *testing.T) {
	const wantErr = "function response id is required to resume an invocation"
	seed := []*session.Event{
		seedUser("inv_a", "first"),
		resumeSeed("inv_a", "root", fnCallContent("fc-a", "tool", nil)),
	}
	withCaller := []runner.RunOption{runner.WithInvocationID("inv-1")}

	t.Run("refused", func(t *testing.T) {
		tests := []iidRefusalCase{
			{name: "resumable_classic_no_caller_id", root: iidClassicRoot, resumable: true},
			{name: "node_root_with_caller_id", root: iidNodeRoot, opts: withCaller},
			{name: "graph_workflow_root_with_caller_id", root: iidGraphRoot, opts: withCaller},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				tc.seed, tc.msg = seed, iidResponses("", "fc-a")
				checkRefused(t, tc, wantErr)
			})
		}
	})

	t.Run("accepted", func(t *testing.T) {
		tests := []struct {
			name      string
			root      iidRootKind
			resumable bool
			msg       *genai.Content
			wantID    string // "" means a minted e-<uuid>
		}{
			{name: "node_root_no_caller_id", root: iidNodeRoot, msg: iidResponses("")},
			{name: "graph_workflow_root_no_caller_id", root: iidGraphRoot, msg: iidResponses("")},
			{name: "second_response_idless_first_matches", root: iidClassicRoot, resumable: true, msg: iidResponses("fc-a", ""), wantID: "inv_a"},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				svc := session.InMemoryService()
				seedResumeSession(t, svc, seed...)
				root := tc.root(t)
				r := newIIDRunner(t, root, svc, tc.resumable)

				events := runTurn(t, r, tc.msg)

				if got := root.runs.Load(); got != 1 {
					t.Errorf("root ran %d times, want 1", got)
				}
				if len(events) == 0 {
					t.Fatal("Run() yielded nothing, want the root's events")
				}
				for _, id := range invocationIDs(events) {
					if tc.wantID == "" && (!strings.HasPrefix(id, "e-") || id == "inv_a") {
						t.Errorf("event invocation ID = %q, want a minted e-<uuid>", id)
					}
					if tc.wantID != "" && id != tc.wantID {
						t.Errorf("event invocation ID = %q, want %q", id, tc.wantID)
					}
				}
			})
		}
	})

	// adk-python runs an LLM-agent root as a node, whose resolution ignores
	// id-less responses (Runner._resolve_invocation_id_from_fr), so the
	// runner stores the message and runs the run callbacks. The agent's
	// contents build then drops the response, because an id-less response
	// answers only an earlier id-less call of its name and the stored call
	// has an id (flows/llm_flows/tools/_rearranger.py,
	// _drop_orphaned_function_responses), drops the call nobody answered,
	// and the model answers.
	t.Run("llm_agent_root_resumable_no_caller_id_reaches_agent", func(t *testing.T) {
		svc := session.InMemoryService()
		seedResumeSession(t, svc, seed...)
		cb := &iidCallbacks{}
		r := newIIDRunner(t, iidLLMRoot(t), svc, true, cb.plugin(t))

		_, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, iidResponses(""), agent.RunConfig{}))

		if len(errs) != 0 {
			t.Errorf("Run() errors = %q, want none", errorTexts(errs))
		}
		if diff := cmp.Diff([]string{"user response tool", "root text llm reply", "root end"}, describeEvents(sessionEvents(t, svc)[len(seed):])); diff != "" {
			t.Errorf("events stored by the run (-want +got):\n%s", diff)
		}
		if got := [3]int32{cb.onUserMessage.Load(), cb.beforeRun.Load(), cb.afterRun.Load()}; got != [3]int32{1, 1, 1} {
			t.Errorf("callbacks [on user message, before run, after run] ran %v times, want [1 1 1]", got)
		}
	})
}

func TestRun_LenientResolution_UnattributableFallsBackToCallerOrMintedID(t *testing.T) {
	messages := []struct {
		name string
		msg  *genai.Content
		// classicOnly marks the id-less message, whose outcome on node,
		// graph and LLM-agent roots depends on the root and the caller's
		// invocation ID: node and graph roots refuse it when the caller
		// passes one, and an LLM-agent root without one accepts it.
		// TestRun_StrictResolution_IDLessFirstFunctionResponse covers both.
		classicOnly bool
		// multiInvocation marks a message a node or graph root refuses on
		// any runner (TestRun_StrictResolution_MultiInvocationOnNodeAndGraphRoots),
		// and that an LLM-agent root without a caller ID runs under the
		// first answered call's invocation.
		multiInvocation bool
	}{
		{name: "unmatched", msg: iidResponses("fc-missing")},
		{name: "id_less", msg: iidResponses(""), classicOnly: true},
		{name: "two_invocations", msg: iidResponses("fc-a", "fc-b"), multiInvocation: true},
		{name: "one_stored_one_unknown", msg: iidResponses("fc-a", "fc-missing")},
	}
	roots := []struct {
		name     string
		root     iidRootKind
		callerID string // "" means none; the run then mints e-<uuid>
		llm      bool
		node     bool // a Config.Node root or a root graph workflow
	}{
		{name: "classic_with_caller_id", root: iidClassicRoot, callerID: "caller-inv"},
		{name: "classic_no_caller_id", root: iidClassicRoot},
		{name: "llm_agent_root_no_caller_id", root: iidLLMRoot, llm: true},
		{name: "node_root_no_caller_id", root: iidNodeRoot, node: true},
		{name: "graph_workflow_root_no_caller_id", root: iidGraphRoot, node: true},
	}
	for _, rt := range roots {
		for _, m := range messages {
			if (rt.llm || rt.node) && m.classicOnly || rt.node && m.multiInvocation {
				continue
			}
			t.Run(rt.name+"/"+m.name, func(t *testing.T) {
				svc := session.InMemoryService()
				seed := iidTwoInvocations()
				seedResumeSession(t, svc, seed...)
				root := rt.root(t)
				r := newIIDRunner(t, root, svc, false)
				var opts []runner.RunOption
				if rt.callerID != "" {
					opts = append(opts, runner.WithInvocationID(rt.callerID))
				}

				events := runTurn(t, r, m.msg, opts...)

				if got := root.runs.Load(); got != 1 {
					t.Errorf("root ran %d times, want 1", got)
				}
				stored := sessionEvents(t, svc)
				if len(stored) <= len(seed) {
					t.Fatalf("session holds %d events, want more than the %d seeded", len(stored), len(seed))
				}
				if len(events) == 0 {
					t.Fatal("Run() yielded nothing, want the root's events")
				}
				ids := append(invocationIDs(stored[len(seed):]), invocationIDs(events)...)
				for _, id := range ids {
					if rt.callerID != "" && id != rt.callerID {
						t.Errorf("event invocation ID = %q, want the caller's %q", id, rt.callerID)
					}
					if rt.llm && m.multiInvocation {
						if id != "inv_a" {
							t.Errorf("event invocation ID = %q, want the first answered call's %q", id, "inv_a")
						}
						continue
					}
					if rt.callerID == "" && !strings.HasPrefix(id, "e-") {
						t.Errorf("event invocation ID = %q, want a minted e-<uuid>", id)
					}
				}
			})
		}
	}
}

// adk-python refuses a nil message on a classic runner that is not resumable
// (runners.py:1274); adk-go runs the root under the caller's ID, as callers
// such as the conformance Adapter's run steps rely on.
func TestRun_NilMessageWithInvocationIDOnNonResumableRunner(t *testing.T) {
	t.Run("classic_nil_msg_with_id_non_resumable", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		root := iidClassicRoot(t)
		r := newIIDRunner(t, root, svc, false)

		events := runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

		if got := root.runs.Load(); got != 1 {
			t.Errorf("root ran %d times, want 1", got)
		}
		if len(events) != 1 || events[0].InvocationID != "inv-1" {
			t.Errorf("Run() yielded %v under %v, want one event under %q", describeEvents(events), invocationIDs(events), "inv-1")
		}
	})

	t.Run("classic_nil_msg_without_id_with_state_delta", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		r := newIIDRunner(t, iidClassicRoot(t), svc, false)

		runTurn(t, r, nil, runner.WithStateDelta(map[string]any{"k": "v"}))

		got, err := storedNodeTestSession(t, t.Context(), svc).State().Get("k")
		if err != nil || got != "v" {
			t.Errorf("State().Get(%q) = %v, %v, want %q, nil", "k", got, err, "v")
		}
	})

	t.Run("node_root_nil_msg_with_id_non_resumable", func(t *testing.T) {
		svc := session.InMemoryService()
		newNodeTestSession(t, t.Context(), svc)
		root := iidNodeRoot(t)
		r := newIIDRunner(t, root, svc, false)

		events := runTurn(t, r, nil, runner.WithInvocationID("inv-1"))

		if len(events) != 1 || events[0].InvocationID != "inv-1" {
			t.Errorf("Run() yielded %v under %v, want one event under %q", describeEvents(events), invocationIDs(events), "inv-1")
		}
	})
}
