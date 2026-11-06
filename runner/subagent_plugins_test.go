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
	"slices"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// pluginRecorder is a plugin that records which of its callbacks fired.
type pluginRecorder struct {
	mu           sync.Mutex
	beforeTools  []string
	beforeAgents []string
	beforeModels []string
	beforeRuns   int
	userMsgs     []string
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
		BeforeAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
			p.mu.Lock()
			defer p.mu.Unlock()
			p.beforeAgents = append(p.beforeAgents, ctx.AgentName())
			return nil, nil
		},
		BeforeModelCallback: func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			p.mu.Lock()
			defer p.mu.Unlock()
			p.beforeModels = append(p.beforeModels, ctx.AgentName())
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
	// The parent's agent and model callbacks fire for the sub-agent too.
	if want := []string{"parent", "sub_agent"}; !slices.Equal(p.beforeAgents, want) {
		t.Errorf("BeforeAgentCallback saw %v, want %v", p.beforeAgents, want)
	}
	if !slices.Contains(p.beforeModels, "sub_agent") {
		t.Errorf("BeforeModelCallback saw %v, want it to include sub_agent", p.beforeModels)
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
// a user message and cannot short-circuit the sub-run from OnUserMessage or
// BeforeRun. Agent, model and tool callbacks still fire for the sub-agent.
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
		t.Errorf("OnUserMessageCallback saw %q, want %q", p.userMsgs, want)
	}
	if p.beforeRuns != 1 {
		t.Errorf("BeforeRunCallback ran %d times, want 1 (the parent run only)", p.beforeRuns)
	}
}

// A runner without plugins that is not started by agenttool must not pick up
// the plugins of the context it is called with. Here a plugin's callback runs
// a judge agent through its own plugin-less runner, passing the callback's
// context. The judge's model call must not fire the callback again.
func TestRunner_PluginCallbackRunnerDoesNotInheritPlugins(t *testing.T) {
	judge, err := llmagent.New(llmagent.Config{
		Name:  "judge",
		Model: &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("SAFE", "model")}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	judgeRunner, err := runner.New(runner.Config{
		AppName:           "judge_app",
		Agent:             judge,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	var mu sync.Mutex
	var seen []string
	pl, err := plugin.New(plugin.Config{
		Name: "llm_judge",
		AfterModelCallback: func(ctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			mu.Lock()
			seen = append(seen, ctx.AgentName())
			n := len(seen)
			mu.Unlock()
			if n > 3 {
				return nil, fmt.Errorf("judge callback re-entered %d times", n)
			}
			for _, err := range judgeRunner.Run(ctx, "judge-user", fmt.Sprintf("judge-%d", n),
				genai.NewContentFromText("is this safe?", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					return nil, err
				}
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}

	parent, err := llmagent.New(llmagent.Config{
		Name:  "parent",
		Model: &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("hello", "model")}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             parent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{pl}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}

	if want := []string{"parent"}; !slices.Equal(seen, want) {
		t.Errorf("AfterModelCallback saw %v, want %v (the judge runner must not inherit the plugin)", seen, want)
	}
}

// A plugin-less runner called with a context that already carries plugins,
// outside agenttool, must not run them.
func TestRunner_SeparateRunnerDoesNotInheritContextPlugins(t *testing.T) {
	p := &pluginRecorder{}
	separateAgent, err := llmagent.New(llmagent.Config{
		Name:  "separate",
		Model: &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("separate done", "model")}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	// The tool below runs under the parent run, so its context carries the
	// parent's plugins when it calls this separate plugin-less runner.
	separate, err := runner.New(runner.Config{
		AppName:           "separate_app",
		Agent:             separateAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	callSeparate, err := functiontool.New(functiontool.Config{Name: "call_separate", Description: "runs another runner"},
		func(ctx agent.Context, _ struct{}) (string, error) {
			for _, err := range separate.Run(ctx, "u2", "s2", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					return "", err
				}
			}
			return "ok", nil
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	parent, err := llmagent.New(llmagent.Config{
		Name: "parent",
		Model: &scriptedModel{responses: []*genai.Content{
			callContent("call_separate", nil),
			genai.NewContentFromText("done", "model"),
		}},
		Tools: []tool.Tool{callSeparate},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
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

	if slices.Contains(p.beforeAgents, "separate") || slices.Contains(p.beforeModels, "separate") {
		t.Errorf("parent plugin fired for the separate runner: agents %v, models %v", p.beforeAgents, p.beforeModels)
	}
}

// The parent's plugin also fires for the agenttool sub-agent. A judge that its
// callback runs from there, through its own plugin-less runner, must not
// inherit the plugin.
func TestRunner_JudgeInsideAgentToolSubRunDoesNotInheritPlugins(t *testing.T) {
	judge, err := llmagent.New(llmagent.Config{
		Name:  "judge",
		Model: &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("SAFE", "model")}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	judgeRunner, err := runner.New(runner.Config{
		AppName:           "judge_app",
		Agent:             judge,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	var mu sync.Mutex
	var seen []string
	pl, err := plugin.New(plugin.Config{
		Name: "llm_judge",
		AfterModelCallback: func(ctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			mu.Lock()
			seen = append(seen, ctx.AgentName())
			n := len(seen)
			mu.Unlock()
			if n > 5 {
				return nil, fmt.Errorf("judge callback re-entered %d times", n)
			}
			for _, err := range judgeRunner.Run(ctx, "judge-user", fmt.Sprintf("judge-%d", n),
				genai.NewContentFromText("is this safe?", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					return nil, err
				}
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}

	sub, err := llmagent.New(llmagent.Config{
		Name:        "sub_agent",
		Description: "does the work",
		Model:       &scriptedModel{responses: []*genai.Content{genai.NewContentFromText("leaf done", "model")}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	parent, err := llmagent.New(llmagent.Config{
		Name: "parent",
		Model: &scriptedModel{responses: []*genai.Content{
			callContent("sub_agent", map[string]any{"request": "delegated prompt"}),
			genai.NewContentFromText("all done", "model"),
		}},
		Tools: []tool.Tool{agenttool.New(sub, nil)},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             parent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{pl}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}

	if slices.Contains(seen, "judge") {
		t.Errorf("AfterModelCallback saw %v, want no judge entries", seen)
	}
}
