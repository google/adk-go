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

package agent_test

import (
	"context"
	"errors"
	"iter"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"
)

// runToEndWithPlugins runs the runner cfg configures, with the given plugins,
// until the run ends.
func runToEndWithPlugins(t *testing.T, cfg runner.Config, plugins ...*plugin.Plugin) {
	t.Helper()
	sessions := session.InMemoryService()
	created, err := sessions.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("session.Create() error = %v", err)
	}
	cfg.AppName = "app"
	cfg.SessionService = sessions
	cfg.PluginConfig = runner.PluginConfig{Plugins: plugins}
	r, err := runner.New(cfg)
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for _, err := range r.Run(t.Context(), "user", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}
}

// A sub-agent's own agent and model callbacks, run after a transfer, see the
// sub-agent. Fails if the callback context's Agent returns nil or the agent of
// an enclosing invocation.
func TestCallbackContextAgent_AgentAndModelCallbacksReportTheirAgent(t *testing.T) {
	var names agentNames
	agentCallback := func(ctx agent.Context) (*genai.Content, error) {
		names.record(ctx)
		return nil, nil
	}
	helper := newTestLLMAgent(t, llmagent.Config{
		Name:                 "helper",
		Model:                &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("done", genai.RoleModel)}},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{agentCallback},
		AfterAgentCallbacks:  []agent.AfterAgentCallback{agentCallback},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			names.record(ctx)
			return nil, nil
		}},
		AfterModelCallbacks: []llmagent.AfterModelCallback{func(ctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			names.record(ctx)
			return nil, nil
		}},
	})
	root := newTestLLMAgent(t, llmagent.Config{
		Name: "coordinator",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			callContent("transfer_to_agent", map[string]any{"agent_name": "helper"}),
		}},
		SubAgents: []agent.Agent{helper},
	})

	runToEnd(t, root)

	// Before agent, before model, after model, after agent.
	if got, want := names.get(), []string{"helper", "helper", "helper", "helper"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the callbacks = %q, want %q", got, want)
	}
}

// An on-model-error callback sees the agent whose model call failed. The mock
// model has no response, so its call fails.
func TestCallbackContextAgent_ModelErrorCallbackReportsTheAgentOfTheCall(t *testing.T) {
	var names agentNames
	a := newTestLLMAgent(t, llmagent.Config{
		Name:  "writer",
		Model: &testutil.MockModel{},
		OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{func(ctx agent.Context, _ *model.LLMRequest, _ error) (*model.LLMResponse, error) {
			names.record(ctx)
			return &model.LLMResponse{Content: genai.NewContentFromText("recovered", genai.RoleModel)}, nil
		}},
	})

	runToEnd(t, a)

	if got, want := names.get(), []string{"writer"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the on-model-error callback = %q, want %q", got, want)
	}
}

// A plugin's agent and model callbacks see the agent they run for.
func TestCallbackContextAgent_PluginCallbacksReportTheAgent(t *testing.T) {
	var names agentNames
	p, err := plugin.New(plugin.Config{
		Name: "observer",
		BeforeAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
			names.record(ctx)
			return nil, nil
		},
		AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
			names.record(ctx)
			return nil, nil
		},
		BeforeModelCallback: func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			names.record(ctx)
			return nil, nil
		},
		AfterModelCallback: func(ctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			names.record(ctx)
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	a := newTestLLMAgent(t, llmagent.Config{
		Name:  "assistant",
		Model: &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("hi", genai.RoleModel)}},
	})

	runToEndWithPlugins(t, runner.Config{Agent: a}, p)

	// Before agent, before model, after model, after agent.
	if got, want := names.get(), []string{"assistant", "assistant", "assistant", "assistant"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the plugin callbacks = %q, want %q", got, want)
	}
}

// A plugin's on-agent-error callback gets the failed agent from its context
// as well as from its argument.
func TestCallbackContextAgent_AgentErrorCallbackReportsTheFailedAgent(t *testing.T) {
	var names agentNames
	p, err := plugin.New(plugin.Config{
		Name: "observer",
		OnAgentErrorCallback: func(ctx agent.Context, _ agent.Agent, _ error) {
			names.record(ctx)
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	failing, err := agent.New(agent.Config{
		Name: "failing",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, errors.New("boom"))
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	sessions := session.InMemoryService()
	created, err := sessions.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("session.Create() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: failing, SessionService: sessions, PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{p}}})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for range r.Run(t.Context(), "user", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) { //nolint:errcheck // the failing agent errors by design; the test reads the callback's names
	}

	if got, want := names.get(), []string{"failing"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the on-agent-error callback = %q, want %q", got, want)
	}
}

// agentToolset records the agent of the context its tools are listed with.
type agentToolset struct {
	names *agentNames
}

func (s *agentToolset) Name() string { return "agent_toolset" }

func (s *agentToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	if c, ok := ctx.(agent.Context); ok {
		s.names.record(c)
	}
	return nil, nil
}

// A toolset listed from a callback, as an application does to see which tools
// a skill offers, gets the callback's agent.
func TestCallbackContextAgent_ToolsetListedFromCallbackReportsTheAgent(t *testing.T) {
	var names agentNames
	ts := &agentToolset{names: &names}
	a, err := agent.New(agent.Config{
		Name: "assistant",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
			_, err := ts.Tools(ctx)
			return nil, err
		}},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	runToEnd(t, a)

	if got, want := names.get(), []string{"assistant"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the toolset listed from the callback = %q, want %q", got, want)
	}
}

// Under a runner whose root is a node, an agent run as a node sees itself in
// its callbacks, and a callback context made from the root's own node context
// sees the runner's stand-in agent, the agent AgentName names there.
func TestCallbackContextAgent_NodeRootReportsTheNodeAgent(t *testing.T) {
	var names agentNames
	writer := newTestLLMAgent(t, llmagent.Config{
		Name:  "writer",
		Model: &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("draft", genai.RoleModel)}},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			names.record(ctx)
			return nil, nil
		}},
	})
	writerNode, err := workflow.NewAgentNode(writer, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode() error = %v", err)
	}
	first := workflow.NewFunctionNode("first", func(ctx agent.Context, in any) (any, error) {
		names.record(agent.NewCallbackContext(ctx, nil))
		return in, nil
	}, workflow.NodeConfig{})
	root, err := workflow.NewWorkflowNode("bare_workflow", workflow.Chain(workflow.Start, first, writerNode))
	if err != nil {
		t.Fatalf("workflow.NewWorkflowNode() error = %v", err)
	}

	runToEndWithPlugins(t, runner.Config{Node: root})

	if got, want := names.get(), []string{"bare_workflow", "writer"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() under the node root = %q, want %q", got, want)
	}
}

// helloExecutor is an A2A agent that answers every message with "hi".
type helloExecutor struct{}

func (helloExecutor) Execute(_ context.Context, _ *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		yield(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("hi")), nil)
	}
}

func (helloExecutor) Cancel(_ context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		yield(a2a.NewStatusUpdateEvent(execCtx, a2a.TaskStateCanceled, nil), nil)
	}
}

// A remote A2A agent's request callbacks see the remote agent.
func TestCallbackContextAgent_RemoteAgentRequestCallbacksReportTheRemoteAgent(t *testing.T) {
	var names agentNames
	server := httptest.NewServer(a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(helloExecutor{})))
	t.Cleanup(server.Close)
	remote, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name: "remote_helper",
		AgentCard: &a2a.AgentCard{
			SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(server.URL, a2a.TransportProtocolJSONRPC)},
		},
		BeforeRequestCallbacks: []remoteagent.BeforeA2ARequestCallback{func(ctx agent.Context, _ *a2a.SendMessageRequest) (*session.Event, error) {
			names.record(ctx)
			return nil, nil
		}},
		AfterRequestCallbacks: []remoteagent.AfterA2ARequestCallback{func(ctx agent.Context, _ *a2a.SendMessageRequest, _ *session.Event, _ error) (*session.Event, error) {
			names.record(ctx)
			return nil, nil
		}},
	})
	if err != nil {
		t.Fatalf("remoteagent.NewA2A() error = %v", err)
	}

	runToEnd(t, remote)

	if got, want := names.get(), []string{"remote_helper", "remote_helper"}; !slices.Equal(got, want) {
		t.Errorf("ctx.Agent().Name() in the request callbacks = %q, want %q", got, want)
	}
}

func TestCallbackContextAgent_PreservesWrapperIdentityAndCanonicalAccessors(t *testing.T) {
	wantModel := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("done", genai.RoleModel)}}
	var helper agent.Agent
	var sawBeforeSame, sawAfterSame bool
	checkLLM := func(same *bool) func(agent.Context) (*genai.Content, error) {
		return func(ctx agent.Context) (*genai.Content, error) {
			*same = ctx.Agent() == helper
			m, err := llmagent.CanonicalModel(ctx.Agent())
			if err != nil {
				return nil, err
			}
			if m != wantModel {
				t.Errorf("CanonicalModel(ctx.Agent()) = %v, want %v", m, wantModel)
			}
			return nil, nil
		}
	}
	helper = newTestLLMAgent(t, llmagent.Config{
		Name:                 "helper",
		Model:                wantModel,
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{checkLLM(&sawBeforeSame)},
		AfterAgentCallbacks:  []agent.AfterAgentCallback{checkLLM(&sawAfterSame)},
	})
	runToEnd(t, helper)
	if !sawBeforeSame || !sawAfterSame {
		t.Errorf("wrapper identity in before/after agent callbacks = (%v, %v), want (true, true)", sawBeforeSame, sawAfterSame)
	}
}
