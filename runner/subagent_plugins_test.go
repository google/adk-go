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
	"slices"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// pluginRecorder is a plugin that records which of its callbacks fired.
type pluginRecorder struct {
	mu          sync.Mutex
	beforeTools []string
	beforeRuns  int
	userMsgs    []string
}

func (p *pluginRecorder) plugin(t *testing.T) *plugin.Plugin {
	t.Helper()
	pl, err := plugin.New(plugin.Config{
		Name: "recorder",
		BeforeToolCallback: func(ctx agent.Context, tl tool.Tool, args map[string]any) (map[string]any, error) {
			p.mu.Lock()
			defer p.mu.Unlock()
			p.beforeTools = append(p.beforeTools, tl.Name())
			return nil, nil
		},
		BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
			p.mu.Lock()
			defer p.mu.Unlock()
			p.beforeRuns++
			return nil, nil
		},
		OnUserMessageCallback: func(_ agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
			p.mu.Lock()
			defer p.mu.Unlock()
			p.userMsgs = append(p.userMsgs, msg.Parts[0].Text)
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return pl
}

func callContent(name string, args map[string]any) *genai.Content {
	return &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: name, Args: args}}}}
}

func newInnerTool(t *testing.T) tool.Tool {
	t.Helper()
	inner, err := functiontool.New(functiontool.Config{Name: "inner_tool", Description: "returns pong"},
		func(_ agent.Context, _ struct{}) (string, error) { return "pong", nil })
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return inner
}

// runParentWithSubAgent runs a parent LlmAgent that calls sub through
// agenttool, with p registered on the parent runner only.
func runParentWithSubAgent(t *testing.T, sub agent.Agent, p *pluginRecorder) {
	t.Helper()
	parentModel := &scriptedModel{responses: []*genai.Content{
		callContent(sub.Name(), map[string]any{"request": "delegated prompt"}),
		genai.NewContentFromText("all done", "model"),
	}}
	parent, err := llmagent.New(llmagent.Config{
		Name:  "parent",
		Model: parentModel,
		Tools: []tool.Tool{agenttool.New(sub, nil)},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "plugins_app",
		Agent:             parent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{p.plugin(t)}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}
}

// A plugin registered on the parent runner must see the tools of an LlmAgent
// that the parent calls through agenttool.
func TestRunner_ParentPluginsReachAgentToolLlmSubAgent(t *testing.T) {
	sub, err := llmagent.New(llmagent.Config{
		Name:        "sub_agent",
		Description: "does the work",
		Model: &scriptedModel{responses: []*genai.Content{
			callContent("inner_tool", nil),
			genai.NewContentFromText("leaf done", "model"),
		}},
		Tools: []tool.Tool{newInnerTool(t)},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	p := &pluginRecorder{}
	runParentWithSubAgent(t, sub, p)

	if want := []string{"sub_agent", "inner_tool"}; !slices.Equal(p.beforeTools, want) {
		t.Errorf("BeforeToolCallback saw %v, want %v", p.beforeTools, want)
	}
}

// Same, with the LlmAgent behind a SequentialAgent wrapper.
func TestRunner_ParentPluginsReachAgentToolSequentialSubAgent(t *testing.T) {
	leaf, err := llmagent.New(llmagent.Config{
		Name: "leaf",
		Model: &scriptedModel{responses: []*genai.Content{
			callContent("inner_tool", nil),
			genai.NewContentFromText("leaf done", "model"),
		}},
		Tools: []tool.Tool{newInnerTool(t)},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "seq_agent", Description: "wraps leaf", SubAgents: []agent.Agent{leaf}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New() error = %v", err)
	}

	p := &pluginRecorder{}
	runParentWithSubAgent(t, seq, p)

	if want := []string{"seq_agent", "inner_tool"}; !slices.Equal(p.beforeTools, want) {
		t.Errorf("BeforeToolCallback saw %v, want %v", p.beforeTools, want)
	}
}

// Inheriting the parent's plugins for tools must not hand the sub-run's
// run-level callbacks to the parent's plugins. Those stay with the runner that
// owns the run, so a guardrail does not see agenttool's synthesized request as
// a user message and cannot short-circuit the sub-agent.
func TestRunner_ParentPluginsDoNotOwnSubRunCallbacks(t *testing.T) {
	sub, err := llmagent.New(llmagent.Config{
		Name:        "sub_agent",
		Description: "does the work",
		Model:       &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("leaf done", "model")}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	p := &pluginRecorder{}
	runParentWithSubAgent(t, sub, p)

	if want := []string{"go"}; !slices.Equal(p.userMsgs, want) {
		t.Errorf("OnUserMessageCallback saw %d messages, want only the end-user message", len(p.userMsgs))
	}
	if p.beforeRuns != 1 {
		t.Errorf("BeforeRunCallback ran %d times, want 1 (the parent run only)", p.beforeRuns)
	}
}
