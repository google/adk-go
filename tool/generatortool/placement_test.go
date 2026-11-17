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
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/generatortool"
	"google.golang.org/adk/v2/workflow"
)

type noArgs struct{}

// newRootHarness is a harness whose runner runs root, an agent that places
// an LLM agent answering from m.
func newRootHarness(t *testing.T, root agent.Agent, m *testutil.MockModel) *harness {
	t.Helper()
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: appName, Agent: root, SessionService: sessions, ArtifactService: artifact.InMemoryService()})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	if _, err := sessions.Create(t.Context(), &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	return &harness{t: t, model: m, runner: r, sessions: sessions}
}

// genAgent is an LLM agent named "a" in mode whose model calls tool gen once
// with id call_1 and then answers "done".
func genAgent(t *testing.T, gen tool.Tool, mode llmagent.Mode) (agent.Agent, *testutil.MockModel) {
	t.Helper()
	m := &testutil.MockModel{Responses: []*genai.Content{calls(call("call_1", "gen", map[string]any{})), modelText("done")}}
	a, err := llmagent.New(llmagent.Config{Name: "a", Model: m, Tools: []tool.Tool{gen}, Mode: mode})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a, m
}

// workflowRoot is a root agent named "wf" that runs a workflow whose only
// node is node.
func workflowRoot(t *testing.T, node workflow.Node) agent.Agent {
	t.Helper()
	wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node))
	if err != nil {
		t.Fatalf("workflow.New() error = %v", err)
	}
	root, err := agent.New(agent.Config{Name: "wf", Run: wf.Run})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return root
}

func agentNode(t *testing.T, a agent.Agent) workflow.Node {
	t.Helper()
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode() error = %v", err)
	}
	return n
}

// hostNode is a dynamic node named "host" that runs a with
// llmagent.RunLLMAgentAsNode on its own context and emits a's events.
func hostNode(a agent.Agent) workflow.Node {
	return workflow.NewDynamicNode("host", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		for ev, err := range llmagent.RunLLMAgentAsNode(a, ctx, nil) {
			if err != nil {
				return nil, err
			}
			if err := emit(ev); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}, workflow.NodeConfig{})
}

// Fails if a generator tool's body does not run, or its output, state write
// or emitted event does not reach the user, in one of the placements an LLM
// agent can have: a workflow's agent node, that node as a dynamic child, a
// dynamic node running the agent in each mode, and a non-LLM root agent
// whose sub-agent runs off the node path.
func TestRun_BodyRunsInEveryPlacement(t *testing.T) {
	tests := []struct {
		name string
		mode llmagent.Mode
		root func(t *testing.T, a agent.Agent) agent.Agent
		// wantGen are the stored events authored by the tool.
		wantGen []evSummary
	}{
		{
			name: "agent node in a workflow",
			root: func(t *testing.T, a agent.Agent) agent.Agent { return workflowRoot(t, agentNode(t, a)) },
		},
		{
			name: "agent node as a dynamic child",
			root: func(t *testing.T, a agent.Agent) agent.Agent {
				child := agentNode(t, a)
				return workflowRoot(t, workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
					return workflow.RunNode[any](ctx, child, nil)
				}, workflow.NodeConfig{}))
			},
		},
		{
			name: "single_turn agent run as a node",
			mode: llmagent.ModeSingleTurn,
			root: func(t *testing.T, a agent.Agent) agent.Agent { return workflowRoot(t, hostNode(a)) },
		},
		{
			name: "undeclared mode agent run as a node",
			root: func(t *testing.T, a agent.Agent) agent.Agent { return workflowRoot(t, hostNode(a)) },
			wantGen: []evSummary{
				{Author: "gen", Branch: "gen@call_1", Path: "wf@1/host@1/gen@call_1", Text: "working"},
				{Author: "gen", Branch: "gen@call_1", Path: "wf@1/host@1/gen@call_1", Output: "ok", NoContent: true},
			},
		},
		{
			name: "chat agent run as a node",
			mode: llmagent.ModeChat,
			root: func(t *testing.T, a agent.Agent) agent.Agent { return workflowRoot(t, hostNode(a)) },
			wantGen: []evSummary{
				{Author: "gen", Branch: "gen@call_1", Path: "wf@1/host@1/gen@call_1", Text: "working"},
				{Author: "gen", Branch: "gen@call_1", Path: "wf@1/host@1/gen@call_1", Output: "ok", NoContent: true},
			},
		},
		{
			name: "sequential agent root",
			root: func(t *testing.T, a agent.Agent) agent.Agent {
				root, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{a}}})
				if err != nil {
					t.Fatalf("sequentialagent.New() error = %v", err)
				}
				return root
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var runs atomic.Int32
			gen := mustNew(t, functiontool.Config{Name: "gen"}, func(ctx agent.Context, _ noArgs, emit func(*session.Event) error) (string, error) {
				runs.Add(1)
				setState(t, ctx, "counter", 7)
				if err := emit(message("working")); err != nil {
					return "", err
				}
				return "ok", nil
			})
			a, m := genAgent(t, gen, tc.mode)
			h := newRootHarness(t, tc.root(t, a), m)

			h.mustRun(userText("go"))

			if got := runs.Load(); got != 1 {
				t.Errorf("handler runs = %d, want 1", got)
			}
			var got []map[string]any
			for _, fr := range responses(h.events(), "gen") {
				got = append(got, fr.Response)
			}
			if diff := cmp.Diff([]map[string]any{{"result": "ok"}}, got); diff != "" {
				t.Errorf("function responses of gen mismatch (-want +got):\n%s", diff)
			}
			if got := h.state("counter"); got != 7 {
				t.Errorf("state counter = %v, want 7", got)
			}
			if diff := cmp.Diff(tc.wantGen, summarize(authoredBy(h.events(), "gen"))); diff != "" {
				t.Errorf("events authored by gen mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// sequentialRoot is a root agent named "seq" whose only sub-agent is a; the
// runner runs it off the node path, so a's tools get no sub-scheduler.
func sequentialRoot(t *testing.T, a agent.Agent) agent.Agent {
	t.Helper()
	root, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{a}}})
	if err != nil {
		t.Fatalf("sequentialagent.New() error = %v", err)
	}
	return root
}

// Fails if a call with no workflow node loses an output, a state change or an
// artifact change the function response event can carry, applies the deltas
// of a partial event, pauses where nothing could answer, or runs the handler
// on another branch than the call's.
func TestRun_OffNodePathKeepsWhatTheResponseCarries(t *testing.T) {
	tests := []struct {
		name          string
		handler       generatortool.Func[noArgs, any]
		wantResponse  map[string]any
		wantState     map[string]any
		wantArtifacts map[string]int64
	}{
		{
			name: "deltas of emitted events",
			handler: func(ctx agent.Context, _ noArgs, emit func(*session.Event) error) (any, error) {
				if err := ctx.State().Set("pending", 1); err != nil {
					return nil, err
				}
				if _, err := ctx.Artifacts().Save(ctx, "a.txt", genai.NewPartFromText("a")); err != nil {
					return nil, err
				}
				own := message("m")
				own.Actions.StateDelta = map[string]any{"own": 2}
				own.Actions.ArtifactDelta = map[string]int64{"own.txt": 3}
				if err := emit(own); err != nil {
					return nil, err
				}
				partial := message("p")
				partial.Partial = true
				partial.Actions.StateDelta = map[string]any{"partial": 3}
				if err := emit(partial); err != nil {
					return nil, err
				}
				if err := ctx.State().Set("late", 4); err != nil {
					return nil, err
				}
				return "ok", nil
			},
			wantResponse:  map[string]any{"result": "ok"},
			wantState:     map[string]any{"pending": 1, "own": 2, "late": 4},
			wantArtifacts: map[string]int64{"a.txt": 1, "own.txt": 3},
		},
		{
			name: "output of an emitted event",
			handler: func(_ agent.Context, _ noArgs, emit func(*session.Event) error) (any, error) {
				if err := emit(&session.Event{Output: "ok"}); err != nil {
					return nil, err
				}
				// A later event without an output keeps the output.
				return nil, emit(message("after"))
			},
			wantResponse:  map[string]any{"result": "ok"},
			wantState:     map[string]any{},
			wantArtifacts: map[string]int64{},
		},
		{
			name: "request for input",
			handler: func(ctx agent.Context, _ noArgs, emit func(*session.Event) error) (any, error) {
				if err := ctx.State().Set("before", 1); err != nil {
					return nil, err
				}
				_, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "ask", Message: "region?"})
				if !errors.Is(err, workflow.ErrInvalidRunNodeContext) {
					return nil, fmt.Errorf("ResumeOrRequestInput() error = %v, want one wrapping workflow.ErrInvalidRunNodeContext", err)
				}
				return nil, err
			},
			wantResponse: map[string]any{
				"error": `generator tool "gen": cannot request input outside a workflow node: workflow: RunNode called outside a dynamic node`,
			},
			wantState:     map[string]any{"before": 1},
			wantArtifacts: map[string]int64{},
		},
		{
			name: "bare node interrupted",
			handler: func(agent.Context, noArgs, func(*session.Event) error) (any, error) {
				return nil, workflow.ErrNodeInterrupted
			},
			wantResponse: map[string]any{
				"error": `generator tool "gen": handler returned workflow.ErrNodeInterrupted without requesting input`,
			},
			wantState:     map[string]any{},
			wantArtifacts: map[string]int64{},
		},
		{
			name: "branch of the call",
			handler: func(ctx agent.Context, _ noArgs, _ func(*session.Event) error) (any, error) {
				return ctx.Branch(), nil
			},
			wantResponse:  map[string]any{"result": "gen@call_1"},
			wantState:     map[string]any{},
			wantArtifacts: map[string]int64{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, m := genAgent(t, mustNew(t, functiontool.Config{Name: "gen"}, tc.handler), "")
			h := newRootHarness(t, sequentialRoot(t, a), m)

			h.mustRun(userText("go"))

			events := h.events()
			var found []*session.Event
			for _, ev := range events {
				if len(responses([]*session.Event{ev}, "gen")) > 0 {
					found = append(found, ev)
				}
			}
			if len(found) != 1 {
				t.Fatalf("events with a function response = %d, want 1; events: %+v", len(found), summarize(events))
			}
			ev := found[0]
			if diff := cmp.Diff(tc.wantResponse, ev.Content.Parts[0].FunctionResponse.Response); diff != "" {
				t.Errorf("function response mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantState, ev.Actions.StateDelta); diff != "" {
				t.Errorf("response event state delta mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantArtifacts, ev.Actions.ArtifactDelta); diff != "" {
				t.Errorf("response event artifact delta mismatch (-want +got):\n%s", diff)
			}
			if got := authoredBy(events, "gen"); len(got) != 0 {
				t.Errorf("events authored by gen = %+v, want none", summarize(got))
			}
			if got := functionCalls(events, workflow.WorkflowInputFunctionCallName); len(got) != 0 {
				t.Errorf("request-input calls = %v, want none", got)
			}
			if got := h.modelRequests(); got != 2 {
				t.Errorf("model requests = %d, want 2", got)
			}
		})
	}
}
