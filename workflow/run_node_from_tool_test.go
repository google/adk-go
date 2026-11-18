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

package workflow_test

import (
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/generatortool"
	"google.golang.org/adk/v2/workflow"
)

// toolTurns runs a root agent over one in-memory session, turn by turn.
type toolTurns struct {
	t         *testing.T
	runner    *runner.Runner
	sessions  session.Service
	sessionID string
}

func newToolTurns(t *testing.T, root agent.Agent) *toolTurns {
	t.Helper()
	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return &toolTurns{t: t, runner: r, sessions: svc, sessionID: created.Session.ID()}
}

// state is the value the stored session holds under key, or nil.
func (h *toolTurns) state(key string) any {
	h.t.Helper()
	got, err := h.sessions.Get(h.t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: h.sessionID})
	if err != nil {
		h.t.Fatalf("session Get() error = %v", err)
	}
	v, _ := got.Session.State().Get(key)
	return v
}

// run sends msg and returns the turn's events; a run error fails the test.
func (h *toolTurns) run(msg *genai.Content, opts ...runner.RunOption) []*session.Event {
	h.t.Helper()
	var events []*session.Event
	for ev, err := range h.runner.Run(h.t.Context(), "u", h.sessionID, msg, agent.RunConfig{}, opts...) {
		if err != nil {
			h.t.Fatalf("Run() error = %v", err)
		}
		events = append(events, ev)
	}
	return events
}

// workflowRootAgent is the workflow agent "wf" whose nodes run in a chain.
func workflowRootAgent(t *testing.T, nodes ...workflow.Node) agent.Agent {
	t.Helper()
	root, err := workflowagent.New(workflowagent.Config{
		Name:  "wf",
		Edges: workflow.Chain(append([]workflow.Node{workflow.Start}, nodes...)...),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	return root
}

// runNodeTool is a function tool named name whose handler runs child with
// workflow.RunNode on the tool's context and returns the child's output.
func runNodeTool(t *testing.T, name string, child workflow.Node) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name}, func(ctx agent.Context, _ struct{}) (any, error) {
		return workflow.RunNode[any](ctx, child, nil)
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return tl
}

// toolCaller is an LLM agent "a" in mode with tools whose model answers from
// responses; onError, when non-nil, is its OnToolErrorCallback.
func toolCaller(t *testing.T, mode llmagent.Mode, m *testutil.MockModel, onError llmagent.OnToolErrorCallback, tools ...tool.Tool) agent.Agent {
	t.Helper()
	cfg := llmagent.Config{Name: "a", Model: m, Tools: tools, Mode: mode}
	if onError != nil {
		cfg.OnToolErrorCallbacks = []llmagent.OnToolErrorCallback{onError}
	}
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a
}

// toolErrors records the errors an OnToolErrorCallback receives.
type toolErrors struct {
	mu   sync.Mutex
	errs []error
}

func (r *toolErrors) callback(_ agent.Context, _ tool.Tool, _ map[string]any, err error) (map[string]any, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.errs = append(r.errs, err)
	return nil, nil
}

func (r *toolErrors) get() []error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]error(nil), r.errs...)
}

func modelCall(name string) *genai.Content {
	return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: name, Args: map[string]any{}}}}}
}

// inputRequests returns the events that request input from the user.
func inputRequests(events []*session.Event) []*session.Event {
	var out []*session.Event
	for _, ev := range events {
		for _, fc := range ev.FunctionCalls() {
			if fc.Name == workflow.WorkflowInputFunctionCallName {
				out = append(out, ev)
				break
			}
		}
	}
	return out
}

// toolResponses returns the events that carry a function response of name.
func toolResponses(events []*session.Event, name string) []*session.Event {
	var out []*session.Event
	for _, ev := range events {
		for _, fr := range ev.FunctionResponses() {
			if fr.Name == name {
				out = append(out, ev)
				break
			}
		}
	}
	return out
}

// lastText is the text of the last event in events that has any.
func lastText(events []*session.Event) string {
	for _, ev := range slices.Backward(events) {
		if c := ev.Content; c != nil {
			var s string
			for _, p := range c.Parts {
				s += p.Text
			}
			if s != "" {
				return s
			}
		}
	}
	return ""
}

// chatHost is a dynamic node "host" that runs a with
// llmagent.RunLLMAgentAsNode and emits its events.
func chatHost(a agent.Agent) workflow.Node {
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

// Report is the typed output of the agent node in the output test.
type Report struct {
	Title string `json:"title"`
}

// Fails if a child that a plain function tool runs with workflow.RunNode, in
// an agent's tool call inside a workflow, does not pause the agent's step for
// input, does not resume it with the answer, loses SkipSummarization, does not
// report a timeout or an output schema failure to the tool error callbacks, or
// becomes the output of the agent node that runs the agent; or if a child that
// the agent's model callback runs with workflow.RunNode loses its state write.
func TestRunNode_FromPlainToolPausesAndPropagatesOptions(t *testing.T) {
	askChild := func() workflow.Node {
		return workflow.NewDynamicNode("ask_child", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
			return workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "q1", Message: "region?"})
		}, workflow.NodeConfig{})
	}

	t.Run("pause_and_resume", func(t *testing.T) {
		t.Run("agent_node_turn_1", func(t *testing.T) {
			m := &testutil.MockModel{Responses: []*genai.Content{modelCall("ask_tool"), genai.NewContentFromText("done", genai.RoleModel)}}
			var toolErrs toolErrors
			a := toolCaller(t, "", m, toolErrs.callback, runNodeTool(t, "ask_tool", askChild()))
			var downstream atomic.Int32
			b := workflow.NewFunctionNode("b", func(_ agent.Context, in any) (any, error) {
				downstream.Add(1)
				return in, nil
			}, workflow.NodeConfig{})
			h := newToolTurns(t, workflowRootAgent(t, agentNodePlacement(t, a), b))

			events := h.run(genai.NewContentFromText("go", genai.RoleUser))

			if got := len(m.Requests); got != 1 {
				t.Errorf("model requests = %d, want 1", got)
			}
			if got := toolErrs.get(); len(got) != 0 {
				t.Errorf("tool error callbacks = %v, want none", got)
			}
			requests := inputRequests(events)
			if len(requests) != 1 {
				t.Fatalf("input requests = %d, want 1", len(requests))
			}
			if diff := cmp.Diff([]string{"q1"}, requests[0].LongRunningToolIDs); diff != "" {
				t.Errorf("LongRunningToolIDs mismatch (-want +got):\n%s", diff)
			}
			if got := downstream.Load(); got != 0 {
				t.Errorf("downstream node runs = %d, want 0", got)
			}
		})

		t.Run("chat_agent_resumes", func(t *testing.T) {
			m := &testutil.MockModel{Responses: []*genai.Content{modelCall("ask_tool"), genai.NewContentFromText("done:EU", genai.RoleModel)}}
			a := toolCaller(t, llmagent.ModeChat, m, nil, runNodeTool(t, "ask_tool", askChild()))
			h := newToolTurns(t, workflowRootAgent(t, chatHost(a)))

			first := h.run(genai.NewContentFromText("go", genai.RoleUser))
			requests := inputRequests(first)
			if len(requests) != 1 || len(m.Requests) != 1 {
				t.Fatalf("turn 1: input requests = %d and model requests = %d, want 1 and 1", len(requests), len(m.Requests))
			}
			answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID: "q1", Name: workflow.WorkflowInputFunctionCallName, Response: map[string]any{"result": "EU"},
			}}}}
			second := h.run(answer, runner.WithInvocationID(first[0].InvocationID))

			if got := len(m.Requests); got != 2 {
				t.Errorf("model requests = %d, want 2", got)
			}
			responses := toolResponses(second, "ask_tool")
			if len(responses) != 1 {
				t.Fatalf("turn 2: function responses of ask_tool = %d, want 1", len(responses))
			}
			if diff := cmp.Diff(map[string]any{"result": "EU"}, responses[0].FunctionResponses()[0].Response); diff != "" {
				t.Errorf("ask_tool response mismatch (-want +got):\n%s", diff)
			}
			if got := lastText(second); got != "done:EU" {
				t.Errorf("final text = %q, want done:EU", got)
			}
		})
	})

	t.Run("skip_summarization", func(t *testing.T) {
		kid := workflow.NewFunctionNode("kid", func(ctx agent.Context, _ any) (string, error) {
			ctx.Actions().SkipSummarization = true
			return "fast", nil
		}, workflow.NodeConfig{})
		m := &testutil.MockModel{Responses: []*genai.Content{modelCall("fast_tool"), genai.NewContentFromText("summary", genai.RoleModel)}}
		a := toolCaller(t, "", m, nil, runNodeTool(t, "fast_tool", kid))
		h := newToolTurns(t, workflowRootAgent(t, agentNodePlacement(t, a)))

		events := h.run(genai.NewContentFromText("go", genai.RoleUser))

		if got := len(m.Requests); got != 1 {
			t.Errorf("model requests = %d, want 1", got)
		}
		responses := toolResponses(events, "fast_tool")
		if len(responses) != 1 {
			t.Fatalf("function responses of fast_tool = %d, want 1", len(responses))
		}
		if !responses[0].Actions.SkipSummarization {
			t.Errorf("function response event SkipSummarization = false, want true")
		}
		if diff := cmp.Diff(map[string]any{"result": "fast"}, responses[0].FunctionResponses()[0].Response); diff != "" {
			t.Errorf("fast_tool response mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("callback_child_state_write_stored", func(t *testing.T) {
		child := workflow.NewFunctionNode("cb_child", func(ctx agent.Context, _ any) (string, error) {
			if err := ctx.State().Set("cb_key", "cb_val"); err != nil {
				return "", err
			}
			return "cb-out", nil
		}, workflow.NodeConfig{})
		var out any
		var runErr error
		m := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("done", genai.RoleModel)}}
		a, err := llmagent.New(llmagent.Config{
			Name:  "a",
			Model: m,
			BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
				out, runErr = workflow.RunNode[any](ctx, child, nil)
				return nil, nil
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		h := newToolTurns(t, workflowRootAgent(t, agentNodePlacement(t, a)))

		h.run(genai.NewContentFromText("go", genai.RoleUser))

		if runErr != nil || out != "cb-out" {
			t.Errorf("RunNode() = %v, %v, want cb-out, nil", out, runErr)
		}
		if got := h.state("cb_key"); got != "cb_val" {
			t.Errorf("stored state cb_key = %v, want cb_val", got)
		}
	})

	// toolError runs child from tool "child_tool" of an agent node and returns
	// the one error the agent's tool error callback received.
	toolError := func(t *testing.T, child workflow.Node) error {
		t.Helper()
		m := &testutil.MockModel{Responses: []*genai.Content{modelCall("child_tool"), genai.NewContentFromText("done", genai.RoleModel)}}
		var toolErrs toolErrors
		a := toolCaller(t, "", m, toolErrs.callback, runNodeTool(t, "child_tool", child))
		newToolTurns(t, workflowRootAgent(t, agentNodePlacement(t, a))).run(genai.NewContentFromText("go", genai.RoleUser))
		errs := toolErrs.get()
		if len(errs) != 1 {
			t.Fatalf("tool error callbacks = %v, want 1", errs)
		}
		return errs[0]
	}

	t.Run("timeout", func(t *testing.T) {
		slow := workflow.NewDynamicNode("slow_child", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}, workflow.NodeConfig{Timeout: 20 * time.Millisecond})

		err := toolError(t, slow)

		te, ok := errors.AsType[*workflow.NodeTimeoutError](err)
		if !ok || te.NodeName != "slow_child" {
			t.Errorf("tool error = %v, want a *workflow.NodeTimeoutError for slow_child", err)
		}
		if !errors.Is(err, workflow.ErrNodeFailed) {
			t.Errorf("tool error = %v, want one wrapping workflow.ErrNodeFailed", err)
		}
	})

	t.Run("output_schema_violation", func(t *testing.T) {
		strict, err := workflow.NewFunctionNodeWithSchema("strict_child", func(agent.Context, any) (any, error) {
			return "not-a-number", nil
		}, nil, &jsonschema.Schema{Type: "integer"}, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("NewFunctionNodeWithSchema() error = %v", err)
		}

		err = toolError(t, strict)

		if !errors.Is(err, workflow.ErrNodeFailed) || !strings.Contains(err.Error(), `output validation failed for node "strict_child"`) {
			t.Errorf("tool error = %v, want an output validation failure of strict_child wrapping workflow.ErrNodeFailed", err)
		}
	})

	t.Run("agent_node_output_ignores_tool_child_output", func(t *testing.T) {
		// runReportNode runs a typed agent node over a as a dynamic child and
		// returns what RunNode returned.
		runReportNode := func(t *testing.T, a agent.Agent) (any, error) {
			t.Helper()
			node, err := workflow.NewAgentNodeTyped[string, Report](a, workflow.NodeConfig{})
			if err != nil {
				t.Fatalf("NewAgentNodeTyped() error = %v", err)
			}
			var out any
			var runErr error
			orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
				out, runErr = workflow.RunNode[any](ctx, node, nil)
				return nil, nil
			}, workflow.NodeConfig{})
			newToolTurns(t, workflowRootAgent(t, orch)).run(genai.NewContentFromText("go", genai.RoleUser))
			return out, runErr
		}

		t.Run("generator_tool_output", func(t *testing.T) {
			gen, err := generatortool.New(functiontool.Config{Name: "gen"}, func(_ agent.Context, _ struct{}, emit func(*session.Event) error) (any, error) {
				return nil, emit(&session.Event{Output: "not a report"})
			})
			if err != nil {
				t.Fatalf("generatortool.New() error = %v", err)
			}
			m := &testutil.MockModel{Responses: []*genai.Content{modelCall("gen"), genai.NewContentFromText(`{"title":"final"}`, genai.RoleModel)}}

			out, err := runReportNode(t, toolCaller(t, "", m, nil, gen))
			if err != nil {
				t.Fatalf("RunNode(agent node) error = %v, want nil", err)
			}
			if diff := cmp.Diff(map[string]any{"title": "final"}, out); diff != "" {
				t.Errorf("RunNode(agent node) output mismatch (-want +got):\n%s", diff)
			}
		})

		t.Run("skipped_summarization", func(t *testing.T) {
			kid := workflow.NewFunctionNode("kid", func(ctx agent.Context, _ any) (string, error) {
				ctx.Actions().SkipSummarization = true
				return "fast", nil
			}, workflow.NodeConfig{})
			m := &testutil.MockModel{Responses: []*genai.Content{modelCall("fast_tool")}}

			out, err := runReportNode(t, toolCaller(t, "", m, nil, runNodeTool(t, "fast_tool", kid)))

			// With the summary skipped the agent gives no answer of its own,
			// so the node has no output; the tool child's "fast" is not it.
			if err != nil || out != nil {
				t.Errorf("RunNode(agent node) = %v, %v, want nil, nil", out, err)
			}
		})
	})
}

// Fails if an agent node that runs a workflow agent, run as a dynamic child,
// does not return the inner workflow's output. The inner workflow stamps its
// events with paths of its own (innerwf@1/leaf@1), not below the agent node's
// path, so they are not the events of a node below the child.
func TestRunNode_AgentNodeOfWorkflowAgentReturnsItsOutput(t *testing.T) {
	leaf := workflow.NewFunctionNode("leaf", func(agent.Context, any) (string, error) {
		return "leaf-out", nil
	}, workflow.NodeConfig{})
	inner, err := workflowagent.New(workflowagent.Config{Name: "innerwf", Edges: workflow.Chain(workflow.Start, leaf)})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	node, err := workflow.NewAgentNode(inner, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode() error = %v", err)
	}
	var out any
	var runErr error
	orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
		out, runErr = workflow.RunNode[any](ctx, node, nil)
		return nil, nil
	}, workflow.NodeConfig{})

	newToolTurns(t, workflowRootAgent(t, orch)).run(genai.NewContentFromText("go", genai.RoleUser))

	if runErr != nil || out != "leaf-out" {
		t.Errorf("RunNode(agent node of a workflow agent) = %v, %v, want leaf-out, nil", out, runErr)
	}
}
