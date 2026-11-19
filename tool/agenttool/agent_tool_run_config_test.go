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

package agenttool_test

import (
	"context"
	"iter"
	"net/http"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
)

// recordingModel answers each call with the next of its responses and keeps
// the config of every request it receives.
type recordingModel struct {
	model.LLM
	name      string
	responses []*genai.Content

	mu      sync.Mutex
	configs []*genai.GenerateContentConfig
}

func (m *recordingModel) Name() string { return m.name }

func (m *recordingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	i := len(m.configs)
	m.configs = append(m.configs, req.Config)
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		content := m.responses[min(i, len(m.responses)-1)]
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// rootBuilder returns the root agent of a run, given the LLM agent that calls
// the agent tool.
type rootBuilder func(t *testing.T, parent agent.Agent) agent.Agent

// rootBuilders covers both ways the runner starts an invocation: an LLM agent
// root, and a root that is not an LLM agent.
var rootBuilders = map[string]rootBuilder{
	"llm_agent_root": func(_ *testing.T, parent agent.Agent) agent.Agent { return parent },
	"sequential_agent_root": func(t *testing.T, parent agent.Agent) agent.Agent {
		root, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{parent}}})
		if err != nil {
			t.Fatalf("sequentialagent.New() error = %v", err)
		}
		return root
	},
}

// runParentWithAgentTool runs a parent agent whose first turn calls an agent
// tool wrapping a child agent, under cfg, and returns the child's model.
func runParentWithAgentTool(t *testing.T, newRoot rootBuilder, cfg agent.RunConfig) *recordingModel {
	t.Helper()
	childModel := &recordingModel{name: "child-model", responses: []*genai.Content{genai.NewContentFromText("child answer", genai.RoleModel)}}
	child, err := llmagent.New(llmagent.Config{Name: "child", Description: "Answers.", Model: childModel})
	if err != nil {
		t.Fatalf("llmagent.New(child) error = %v", err)
	}
	parentModel := &recordingModel{name: "parent-model", responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "child", Args: map[string]any{"request": "go"}}}}},
		genai.NewContentFromText("parent answer", genai.RoleModel),
	}}
	parent, err := llmagent.New(llmagent.Config{
		Name:  "parent",
		Model: parentModel,
		Tools: []tool.Tool{agenttool.New(child, nil)},
	})
	if err != nil {
		t.Fatalf("llmagent.New(parent) error = %v", err)
	}
	runner := testutil.NewTestAgentRunner(t, newRoot(t, parent))
	if _, err := testutil.CollectEvents(runner.RunContentWithConfig(t, "session", genai.NewContentFromText("hi", genai.RoleUser), cfg)); err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if len(childModel.configs) == 0 {
		t.Fatal("the child agent's model got no request")
	}
	return childModel
}

// TestAgentTool_Run_ForwardsRunConfigHTTPOptions checks that the nested run
// of an agent tool applies the caller's RunConfig.HTTPOptions to the child
// agent's model requests, as adk-python's AgentTool forwards the caller's run
// config (tools/agent_tool.py).
func TestAgentTool_Run_ForwardsRunConfigHTTPOptions(t *testing.T) {
	for name, newRoot := range rootBuilders {
		t.Run(name, func(t *testing.T) {
			childModel := runParentWithAgentTool(t, newRoot, agent.RunConfig{
				HTTPOptions: &genai.HTTPOptions{Headers: http.Header{"X-Req": {"r1"}}},
			})

			cfg := childModel.configs[0]
			if cfg == nil || cfg.HTTPOptions == nil {
				t.Fatalf("child request config = %+v, want HTTPOptions", cfg)
			}
			if got := cfg.HTTPOptions.Headers.Get("X-Req"); got != "r1" {
				t.Errorf("child request header X-Req = %q, want %q", got, "r1")
			}
		})
	}
}

// TestAgentTool_Run_ForwardsRunConfigLabels checks that the nested run of an
// agent tool applies the caller's RunConfig.Labels to the child agent's model
// requests, next to the child's own agent-name label, as adk-python's
// AgentTool forwards the caller's run config (tools/agent_tool.py).
func TestAgentTool_Run_ForwardsRunConfigLabels(t *testing.T) {
	for name, newRoot := range rootBuilders {
		t.Run(name, func(t *testing.T) {
			labels := map[string]string{"customer": "acme"}
			childModel := runParentWithAgentTool(t, newRoot, agent.RunConfig{Labels: &labels})

			cfg := childModel.configs[0]
			if cfg == nil {
				t.Fatal("child request config is nil, want labels")
			}
			want := map[string]string{"customer": "acme", "adk_agent_name": "child"}
			if diff := cmp.Diff(want, cfg.Labels); diff != "" {
				t.Errorf("child request labels mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAgentTool_Run_ForwardsRunConfigCustomMetadata(t *testing.T) {
	for name, newRoot := range rootBuilders {
		t.Run(name, func(t *testing.T) {
			var childSeen map[string]any
			childModel := &recordingModel{name: "child-model", responses: []*genai.Content{genai.NewContentFromText("child answer", genai.RoleModel)}}
			child, err := llmagent.New(llmagent.Config{
				Name:        "child",
				Description: "Answers.",
				Model:       childModel,
				BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
					childSeen = agent.CustomMetadata(ctx)
					return nil, nil
				}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(child) error = %v", err)
			}
			parentModel := &recordingModel{name: "parent-model", responses: []*genai.Content{
				{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "child", Args: map[string]any{"request": "go"}}}}},
				genai.NewContentFromText("parent answer", genai.RoleModel),
			}}
			parent, err := llmagent.New(llmagent.Config{
				Name:  "parent",
				Model: parentModel,
				Tools: []tool.Tool{agenttool.New(child, nil)},
			})
			if err != nil {
				t.Fatalf("llmagent.New(parent) error = %v", err)
			}
			r := testutil.NewTestAgentRunner(t, newRoot(t, parent))
			meta := map[string]any{"tier": "gold", "__adk_internal_forged": true}
			if _, err := testutil.CollectEvents(r.RunContentWithConfig(t, "session", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{
				CustomMetadata: &meta,
			})); err != nil {
				t.Fatalf("run failed: %v", err)
			}
			want := map[string]any{"tier": "gold"}
			if diff := cmp.Diff(want, childSeen); diff != "" {
				t.Errorf("child agent.CustomMetadata(ctx) mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
